"""Evaluation metrics, bootstrap confidence intervals, and stratified scoring.

Pipeline position: **TRAIN → EVALUATE → REPORT**

Every experiment in this project reports AUC on ``test_real`` — the 1,318 real
rows held out before amplification. Synthetic test data is used only for scale
experiments (E6) and as a sanity check (E7), never as a headline number.

Why bootstrap instead of a normal-approximation CI
--------------------------------------------------
AUC is a ratio of concordant to total pairs, so its sampling distribution is
skewed near 1.0 and the normal approximation understates the upper tail. The
bootstrap makes no distributional assumption and costs nothing at this scale
(1,000 resamples of 1,318 rows takes well under a second).

Usage
-----
>>> from src.evaluation.s12_eval import evaluate_model, bootstrap_ci
>>> metrics = evaluate_model(model, X_test, y_test)
>>> metrics["auc"]
0.89...
>>> bootstrap_ci(y_test, y_score, roc_auc_score)
(0.86..., 0.92...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

# Fraction of the population flagged for the retention campaign
RECALL_TOP_FRACTION: float = 0.10

# Bootstrap settings
DEFAULT_N_RESAMPLES: int = 1000
DEFAULT_ALPHA: float = 0.05


# =============================================================================
# Core metrics
# =============================================================================
def recall_at_top_fraction(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = RECALL_TOP_FRACTION,
) -> float:
    """Share of churners captured by flagging the top ``fraction`` of scores.

    This is the metric a retention team actually cares about: "if we call the
    riskiest 10% of customers, how many churners do we reach?"
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    n_top = max(1, int(round(n * fraction)))
    order = np.argsort(-y_score)
    top = y_true[order[:n_top]]
    total_positives = int(y_true.sum())
    if total_positives == 0:
        return float("nan")
    return float(top.sum() / total_positives)


def predict_proba(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Return churn probabilities for a feature matrix."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float()
        proba = model(x).squeeze(1).cpu().numpy()
    return proba


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute AUC, PR-AUC, F1, and Recall@10% for a model on a labelled set.

    Parameters
    ----------
    model:
        Any module returning a probability in ``(0, 1)``.
    X, y:
        Features and labels. ``X`` must already be scaled the same way the
        model was trained.
    threshold:
        Decision threshold for F1. AUC, PR-AUC, and Recall@10% are
        threshold-free.

    Returns
    -------
    dict
        ``{auc, pr_auc, f1, recall_at_10pct, n_samples, n_churned}``

    Notes
    -----
    ``pr_auc`` (average precision) is the metric to read on this dataset. At a
    28% churn rate a model can be well calibrated below 0.5 everywhere and
    still rank perfectly, which makes ``f1`` at the default threshold collapse
    to 0 even though the model is fine. PR-AUC has no threshold, so it measures
    ranking quality on the minority class directly.
    """
    y_score = predict_proba(model, X)
    y_true = np.asarray(y)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    recall10 = recall_at_top_fraction(y_true, y_score)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "recall_at_10pct": recall10,
        "n_samples": int(len(y_true)),
        "n_churned": int(y_true.sum()),
    }


# =============================================================================
# Bootstrap confidence intervals
# =============================================================================
def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric.

    Parameters
    ----------
    y_true, y_score:
        Labels and scores of the evaluation set.
    metric_fn:
        Callable ``(y_true, y_score) -> float``. Defaults to ``roc_auc_score``.
    n_resamples:
        Number of bootstrap draws.
    alpha:
        ``0.05`` gives a 95% interval.
    seed:
        RNG seed so the interval is reproducible.

    Returns
    -------
    (lower, upper)
        Percentile bounds. Returns ``(nan, nan)`` when a resample contains only
        one class, which makes the metric undefined.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(float(metric_fn(yt, y_score[idx])))
        except ValueError:
            continue

    if not values:
        return float("nan"), float("nan")

    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate_with_ci(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a model and attach bootstrap CIs for AUC, PR-AUC, and Recall@10%."""
    metrics = evaluate_model(model, X, y)
    y_score = predict_proba(model, X)
    y_true = np.asarray(y)

    auc_lo, auc_hi = bootstrap_ci(
        y_true, y_score, roc_auc_score, n_resamples=n_resamples, seed=seed
    )
    pr_lo, pr_hi = bootstrap_ci(
        y_true, y_score, average_precision_score, n_resamples=n_resamples, seed=seed
    )
    rec_lo, rec_hi = bootstrap_ci(
        y_true,
        y_score,
        lambda yt, ys: recall_at_top_fraction(yt, ys),
        n_resamples=n_resamples,
        seed=seed,
    )

    metrics.update({
        "auc_ci_lower": auc_lo,
        "auc_ci_upper": auc_hi,
        "pr_auc_ci_lower": pr_lo,
        "pr_auc_ci_upper": pr_hi,
        "recall_at_10pct_ci_lower": rec_lo,
        "recall_at_10pct_ci_upper": rec_hi,
    })
    return metrics


# =============================================================================
# Per-client evaluation (E5.1)
# =============================================================================
def evaluate_per_client(
    model: torch.nn.Module,
    partition_dir: str | Path,
    vocab: Any,
    k: int,
    *,
    use_local_scaler: bool = True,
) -> pd.DataFrame:
    """Score the global model on each silo separately.

    Parameters
    ----------
    use_local_scaler:
        When ``True`` (default) each silo is scaled with its own statistics,
        which is what the client saw during training. Set to ``False`` to use
        the canonical ``train_real`` scaler instead.

    Returns
    -------
    pandas.DataFrame
        One row per client with ``client_id, n_rows, n_churned, churn_rate,
        auc, f1, recall_at_10pct``.
    """
    from src.features.s07_preprocess import load_client

    rows: list[dict[str, Any]] = []
    for cid in range(k):
        X, y, _ = load_client(partition_dir, cid, vocab, fit_scaler=use_local_scaler)
        metrics = evaluate_model(model, X, y)
        rows.append({
            "client_id": cid,
            "n_rows": metrics["n_samples"],
            "n_churned": metrics["n_churned"],
            "churn_rate": float(y.mean()),
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "recall_at_10pct": metrics["recall_at_10pct"],
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Per-client AUC: mean=%.4f std=%.4f min=%.4f max=%.4f",
        df["auc"].mean(),
        df["auc"].std(),
        df["auc"].min(),
        df["auc"].max(),
    )
    return df


# =============================================================================
# Per-stratum evaluation (E5.2)
# =============================================================================
def evaluate_per_stratum(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    *,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Score the model separately within each stratum.

    Used for E5.2, where strata are the ``Churn Category`` values joined back
    from ``eda_7043.parquet``. Strata smaller than ``min_samples`` are reported
    with ``auc = nan`` because AUC on a handful of rows is meaningless.

    Returns
    -------
    pandas.DataFrame
        One row per stratum with ``stratum, n_rows, n_churned, auc``.
    """
    y_score = predict_proba(model, X)
    y_true = np.asarray(y)
    strata_arr = np.asarray(strata)

    rows: list[dict[str, Any]] = []
    for value in pd.unique(strata_arr):
        mask = strata_arr == value
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())

        if n < min_samples or n_pos == 0 or n_pos == n:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true[mask], y_score[mask]))

        rows.append({
            "stratum": value,
            "n_rows": n,
            "n_churned": n_pos,
            "auc": auc,
        })

    df = pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    logger.info("Per-stratum evaluation: %s strata", len(df))
    return df


# =============================================================================
# Per-reason recall (E5.2, correct formulation)
# =============================================================================
def evaluate_recall_by_reason(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    reasons: np.ndarray,
    *,
    fraction: float = RECALL_TOP_FRACTION,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Recall within each churn reason, using one global threshold.

    ``Churn Category`` is only populated for customers who actually churned, so
    every reason stratum is 100% positive. AUC is undefined there — there is no
    negative class to rank against. The meaningful question is instead: *of the
    customers who left for reason R, what share did the model flag?*

    The top-``fraction`` cut-off is computed **once over the whole test set**,
    not per stratum. A per-stratum cut-off would flag the same share of every
    reason by construction and hide exactly the disparity this measures.

    Parameters
    ----------
    reasons:
        Churn reason per row. Non-churners carry ``"None"`` and are excluded,
        because recall is only defined for actual churners.
    fraction:
        Share of the full test set flagged as high risk.

    Returns
    -------
    pandas.DataFrame
        One row per reason with ``reason, n_churned, n_flagged, recall``.
    """
    y_score = predict_proba(model, X)
    y_true = np.asarray(y)
    reasons_arr = np.asarray(reasons)

    # One global cut-off: the top `fraction` of the whole test set by score
    n_top = max(1, int(round(len(y_true) * fraction)))
    threshold = float(np.sort(y_score)[-n_top])

    rows: list[dict[str, Any]] = []
    for value in pd.unique(reasons_arr):
        if value == "None":
            continue
        mask = reasons_arr == value
        n = int(mask.sum())
        if n < min_samples:
            continue
        flagged = int((y_score[mask] >= threshold).sum())
        rows.append({
            "reason": value,
            "n_churned": n,
            "n_flagged": flagged,
            "recall": flagged / n,
        })

    df = pd.DataFrame(rows, columns=["reason", "n_churned", "n_flagged", "recall"])
    if not df.empty:
        df = df.sort_values("n_churned", ascending=False).reset_index(drop=True)
        logger.info(
            "Recall by reason: mean=%.4f std=%.4f min=%.4f max=%.4f",
            df["recall"].mean(),
            df["recall"].std(),
            df["recall"].min(),
            df["recall"].max(),
        )
    return df


# =============================================================================
# Comparison table
# =============================================================================
def build_comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Turn ``{model_name: metrics}`` into a tidy comparison table.

    Adds a ``gap_vs_centralized_pct`` column so the E1 headline number is
    computed once, in one place, rather than in the report.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "auc": metrics.get("auc", float("nan")),
            "auc_ci_lower": metrics.get("auc_ci_lower", float("nan")),
            "auc_ci_upper": metrics.get("auc_ci_upper", float("nan")),
            "f1": metrics.get("f1", float("nan")),
            "recall_at_10pct": metrics.get("recall_at_10pct", float("nan")),
        })

    df = pd.DataFrame(rows)

    central = df.loc[df["model"] == "centralized", "auc"]
    if not central.empty and float(central.iloc[0]) > 0:
        base = float(central.iloc[0])
        df["gap_vs_centralized_pct"] = (base - df["auc"]) / base * 100
    else:
        df["gap_vs_centralized_pct"] = float("nan")

    return df


def main() -> None:
    """CLI entry point: ``python -m src.evaluation.s12_eval``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.config import get_config
    from src.features.s07_preprocess import fit_vocabulary, load_client
    from src.models.s08_mlp import build_model

    cfg = get_config()
    part_dir = cfg.paths.synthetic_dir / "partitions_iid"
    vocab = fit_vocabulary(part_dir, cfg.partitioning.n_clients)
    X, y, _ = load_client(part_dir, 0, vocab)

    model = build_model(X.shape[1], seed=cfg.project.seed)
    metrics = evaluate_with_ci(model, X[:5000], y[:5000])

    print("\n" + "=" * 70)
    print("EVALUATION SMOKE TEST (untrained model, 5k rows)")
    print("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:28}: {value:.4f}")
        else:
            print(f"  {key:28}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
