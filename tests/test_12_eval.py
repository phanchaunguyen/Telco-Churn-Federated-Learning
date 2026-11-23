"""Tests for :mod:`src.evaluation.s12_eval`.

Run with::

    pytest tests/test_12_eval.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.evaluation.s12_eval import (
    bootstrap_ci,
    build_comparison_table,
    evaluate_model,
    evaluate_per_stratum,
    evaluate_recall_by_reason,
    evaluate_with_ci,
    predict_proba,
    recall_at_top_fraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _FixedModel(torch.nn.Module):
    """Returns a pre-set score per row, so metrics are exactly predictable."""

    def __init__(self, scores: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scores[: len(x)].reshape(-1, 1)


@pytest.fixture
def perfect_data():
    """Scores perfectly separate the classes."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y, scores


@pytest.fixture
def random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    scores = rng.random(500)
    return y, scores


# ---------------------------------------------------------------------------
# Recall@10%
# ---------------------------------------------------------------------------
def test_recall_at_10pct_perfect_ranking() -> None:
    """If all churners rank first, recall@10% is capped by the budget."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # Top 10% of 10 rows = 1 row, which is a churner -> 1 of 2 churners
    assert recall_at_top_fraction(y, scores) == pytest.approx(0.5)


def test_recall_at_10pct_all_churners_in_top() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # Top 10% = 1 row, only 1 churner captured of 2
    assert recall_at_top_fraction(y, scores) == pytest.approx(0.5)


def test_recall_at_10pct_no_positives() -> None:
    y = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    assert np.isnan(recall_at_top_fraction(y, scores))


def test_recall_at_10pct_empty() -> None:
    assert np.isnan(recall_at_top_fraction(np.array([]), np.array([])))


def test_recall_at_10pct_budget_rounds_up() -> None:
    """A 5-row set must still flag at least one row."""
    y = np.array([1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    assert recall_at_top_fraction(y, scores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------
def test_evaluate_model_perfect(perfect_data) -> None:
    y, scores = perfect_data
    model = _FixedModel(scores)
    metrics = evaluate_model(model, np.zeros((len(y), 1)), y)

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["n_samples"] == len(y)
    assert metrics["n_churned"] == int(y.sum())


def test_evaluate_model_random_near_half(random_data) -> None:
    y, scores = random_data
    model = _FixedModel(scores)
    metrics = evaluate_model(model, np.zeros((len(y), 1)), y)
    assert 0.35 < metrics["auc"] < 0.65


def test_evaluate_model_single_class_returns_nan() -> None:
    y = np.zeros(20, dtype=int)
    model = _FixedModel(np.linspace(0, 1, 20))
    metrics = evaluate_model(model, np.zeros((20, 1)), y)
    assert np.isnan(metrics["auc"])


def test_evaluate_model_f1_respects_threshold(perfect_data) -> None:
    y, scores = perfect_data
    model = _FixedModel(scores)
    metrics = evaluate_model(model, np.zeros((len(y), 1)), y, threshold=0.5)
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_model_returns_all_keys(perfect_data) -> None:
    y, scores = perfect_data
    metrics = evaluate_model(_FixedModel(scores), np.zeros((len(y), 1)), y)
    assert set(metrics) == {
        "auc",
        "pr_auc",
        "f1",
        "recall_at_10pct",
        "n_samples",
        "n_churned",
    }


def test_evaluate_model_pr_auc_is_one_for_perfect_ranking(perfect_data) -> None:
    """Perfect separation must give PR-AUC 1.0, same as AUC."""
    y, scores = perfect_data
    metrics = evaluate_model(_FixedModel(scores), np.zeros((len(y), 1)), y)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_evaluate_model_pr_auc_beats_f1_when_scores_are_low() -> None:
    """PR-AUC must stay high when every score sits below the F1 threshold.

    This is the regression that motivated adding PR-AUC: at a 28% churn rate a
    well-ranked model can output probabilities below 0.5 everywhere, so F1 at
    the default threshold collapses to 0 while the ranking is still perfect.
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09])
    metrics = evaluate_model(_FixedModel(scores), np.zeros((len(y), 1)), y)

    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------
def test_predict_proba_shape(perfect_data) -> None:
    y, scores = perfect_data
    proba = predict_proba(_FixedModel(scores), np.zeros((len(y), 1)))
    assert proba.shape == (len(y),)


def test_predict_proba_sets_eval_mode() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Sigmoid())
    model.train()
    predict_proba(model, np.zeros((4, 3)))
    assert not model.training


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_point_estimate(random_data) -> None:
    y, scores = random_data
    point = roc_auc_score(y, scores)
    lo, hi = bootstrap_ci(y, scores, roc_auc_score, n_resamples=300)
    assert lo < point < hi


def test_bootstrap_ci_is_reproducible(random_data) -> None:
    y, scores = random_data
    a = bootstrap_ci(y, scores, roc_auc_score, n_resamples=200, seed=7)
    b = bootstrap_ci(y, scores, roc_auc_score, n_resamples=200, seed=7)
    assert a == b


def test_bootstrap_ci_different_seed_differs(random_data) -> None:
    y, scores = random_data
    a = bootstrap_ci(y, scores, roc_auc_score, n_resamples=200, seed=1)
    b = bootstrap_ci(y, scores, roc_auc_score, n_resamples=200, seed=2)
    assert a != b


def test_bootstrap_ci_wider_at_higher_confidence(random_data) -> None:
    y, scores = random_data
    lo95, hi95 = bootstrap_ci(y, scores, roc_auc_score, n_resamples=500, alpha=0.05)
    lo80, hi80 = bootstrap_ci(y, scores, roc_auc_score, n_resamples=500, alpha=0.20)
    assert (hi95 - lo95) > (hi80 - lo80)


def test_bootstrap_ci_empty_returns_nan() -> None:
    lo, hi = bootstrap_ci(np.array([]), np.array([]), roc_auc_score)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_single_class_returns_nan() -> None:
    y = np.zeros(50, dtype=int)
    scores = np.linspace(0, 1, 50)
    lo, hi = bootstrap_ci(y, scores, roc_auc_score, n_resamples=50)
    assert np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# evaluate_with_ci
# ---------------------------------------------------------------------------
def test_evaluate_with_ci_adds_bounds(random_data) -> None:
    y, scores = random_data
    metrics = evaluate_with_ci(_FixedModel(scores), np.zeros((len(y), 1)), y, n_resamples=200)

    assert "auc_ci_lower" in metrics
    assert "auc_ci_upper" in metrics
    assert metrics["auc_ci_lower"] <= metrics["auc"] <= metrics["auc_ci_upper"]


def test_evaluate_with_ci_recall_bounds(random_data) -> None:
    y, scores = random_data
    metrics = evaluate_with_ci(_FixedModel(scores), np.zeros((len(y), 1)), y, n_resamples=200)
    assert metrics["recall_at_10pct_ci_lower"] <= metrics["recall_at_10pct_ci_upper"]


# ---------------------------------------------------------------------------
# Per-stratum
# ---------------------------------------------------------------------------
def test_evaluate_per_stratum_splits_correctly() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    scores = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    strata = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)

    df = evaluate_per_stratum(_FixedModel(scores), np.zeros((n, 1)), y, strata)
    assert len(df) == 3
    assert set(df["stratum"]) == {"A", "B", "C"}
    assert df["n_rows"].sum() == n


def test_evaluate_per_stratum_small_stratum_gets_nan() -> None:
    y = np.array([0, 1] * 50)
    scores = np.linspace(0, 1, 100)
    strata = np.array(["big"] * 90 + ["tiny"] * 10)

    df = evaluate_per_stratum(
        _FixedModel(scores), np.zeros((100, 1)), y, strata, min_samples=30
    )
    tiny = df.loc[df["stratum"] == "tiny", "auc"].iloc[0]
    assert np.isnan(tiny)


def test_evaluate_per_stratum_single_class_gets_nan() -> None:
    y = np.array([0] * 50 + [0, 1] * 25)
    scores = np.linspace(0, 1, 100)
    strata = np.array(["all_neg"] * 50 + ["mixed"] * 50)

    df = evaluate_per_stratum(_FixedModel(scores), np.zeros((100, 1)), y, strata)
    all_neg = df.loc[df["stratum"] == "all_neg", "auc"].iloc[0]
    assert np.isnan(all_neg)


# ---------------------------------------------------------------------------
# Recall by churn reason (E5.2)
# ---------------------------------------------------------------------------
def test_recall_by_reason_uses_one_global_threshold() -> None:
    """The cut-off must come from the whole test set, not from each reason.

    A per-reason cut-off would flag the same share of every reason by
    construction and hide the disparity this metric exists to measure.
    """
    # 200 rows: 100 churners split across two reasons, 100 non-churners.
    # Scores are set so the global top-10% (20 rows) are all reason "easy".
    y = np.array([1] * 100 + [0] * 100)
    scores = np.concatenate([
        np.linspace(0.90, 0.99, 50),   # reason "easy"  -> highest scores
        np.linspace(0.10, 0.20, 50),   # reason "hard"  -> lowest scores
        np.linspace(0.00, 0.05, 100),  # non-churners
    ])
    reasons = np.array(["easy"] * 50 + ["hard"] * 50 + ["None"] * 100)

    df = evaluate_recall_by_reason(
        _FixedModel(scores), np.zeros((200, 1)), y, reasons, fraction=0.10
    )

    easy = df.loc[df["reason"] == "easy", "recall"].iloc[0]
    hard = df.loc[df["reason"] == "hard", "recall"].iloc[0]

    # Top 20 of 200 rows are all "easy" -> recall 20/50, "hard" gets nothing
    assert easy == pytest.approx(20 / 50)
    assert hard == pytest.approx(0.0)


def test_recall_by_reason_excludes_none_stratum() -> None:
    """Non-churners are not a churn reason and must not appear in the table."""
    y = np.array([1] * 40 + [0] * 60)
    scores = np.linspace(0, 1, 100)
    reasons = np.array(["Price"] * 40 + ["None"] * 60)

    df = evaluate_recall_by_reason(_FixedModel(scores), np.zeros((100, 1)), y, reasons)

    assert "None" not in set(df["reason"])
    assert set(df["reason"]) == {"Price"}


def test_recall_by_reason_drops_small_reasons() -> None:
    """A reason with fewer than min_samples churners is dropped, not reported."""
    y = np.array([1] * 50 + [0] * 50)
    scores = np.linspace(0, 1, 100)
    reasons = np.array(["big"] * 45 + ["tiny"] * 5 + ["None"] * 50)

    df = evaluate_recall_by_reason(
        _FixedModel(scores), np.zeros((100, 1)), y, reasons, min_samples=30
    )

    assert set(df["reason"]) == {"big"}


def test_recall_by_reason_counts_flagged_rows() -> None:
    """n_flagged must equal the number of rows at or above the global cut-off."""
    # Price rows carry the highest scores, so the global top-10% is all Price
    y = np.array([1] * 50 + [0] * 50)
    scores = np.concatenate([np.linspace(0.90, 1.00, 50), np.linspace(0.0, 0.5, 50)])
    reasons = np.array(["Price"] * 50 + ["None"] * 50)

    df = evaluate_recall_by_reason(
        _FixedModel(scores), np.zeros((100, 1)), y, reasons, fraction=0.10
    )

    row = df.iloc[0]
    assert row["n_churned"] == 50
    assert row["n_flagged"] == 10
    assert row["recall"] == pytest.approx(0.2)


def test_recall_by_reason_empty_when_all_none() -> None:
    """With no churn reasons at all the table is empty, not an error."""
    y = np.zeros(100, dtype=int)
    scores = np.linspace(0, 1, 100)
    reasons = np.array(["None"] * 100)

    df = evaluate_recall_by_reason(_FixedModel(scores), np.zeros((100, 1)), y, reasons)

    assert df.empty
    assert list(df.columns) == ["reason", "n_churned", "n_flagged", "recall"]


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def test_comparison_table_computes_gap() -> None:
    results = {
        "centralized": {"auc": 0.90, "f1": 0.6, "recall_at_10pct": 0.4},
        "fl": {"auc": 0.87, "f1": 0.58, "recall_at_10pct": 0.38},
        "local_only": {"auc": 0.80, "f1": 0.5, "recall_at_10pct": 0.3},
    }
    df = build_comparison_table(results)

    fl_gap = df.loc[df["model"] == "fl", "gap_vs_centralized_pct"].iloc[0]
    assert fl_gap == pytest.approx((0.90 - 0.87) / 0.90 * 100)

    central_gap = df.loc[df["model"] == "centralized", "gap_vs_centralized_pct"].iloc[0]
    assert central_gap == pytest.approx(0.0)


def test_comparison_table_without_centralized() -> None:
    df = build_comparison_table({"fl": {"auc": 0.87}})
    assert np.isnan(df["gap_vs_centralized_pct"].iloc[0])


def test_comparison_table_handles_missing_ci() -> None:
    df = build_comparison_table({"fl": {"auc": 0.87}})
    assert np.isnan(df["auc_ci_lower"].iloc[0])
