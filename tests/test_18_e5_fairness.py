"""Tests for :mod:`src.experiments.s18_e5_fairness`.

The full E5 run takes ~15 minutes, so only the pure helpers are tested here.

Run with::

    pytest tests/test_18_e5_fairness.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.s18_e5_fairness import (
    MIN_STRATUM_SAMPLES,
    _render_e5,
    load_strata,
)


# ---------------------------------------------------------------------------
# load_strata
# ---------------------------------------------------------------------------
def test_load_strata_aligns_to_test_frame(monkeypatch) -> None:
    """Strata must come back in the same row order as ``test_df``."""
    test_df = pd.DataFrame({"Customer ID": ["c", "a", "b"]})
    eda = pd.DataFrame({
        "Customer ID": ["a", "b", "c"],
        "Churn Category": ["Price", "Competitor", "Attitude"],
    })

    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: eda)
    strata = load_strata(test_df)

    assert list(strata) == ["Attitude", "Price", "Competitor"]


def test_load_strata_raises_on_missing_ids(monkeypatch) -> None:
    """A missing join key would silently misalign strata, so it must raise."""
    test_df = pd.DataFrame({"Customer ID": ["a", "zzz"]})
    eda = pd.DataFrame({"Customer ID": ["a"], "Churn Category": ["Price"]})

    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: eda)
    with pytest.raises(KeyError, match="no match for Customer ID"):
        load_strata(test_df)


def test_load_strata_normalizes_na_category_to_none(monkeypatch) -> None:
    """Non-churners have Category NA at source; they must become stratum 'None'."""
    test_df = pd.DataFrame({"Customer ID": ["a", "b"]})
    eda = pd.DataFrame({
        "Customer ID": ["a", "b"],
        "Churn Category": [np.nan, "Price"],
    })

    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: eda)
    strata = load_strata(test_df)

    assert list(strata) == ["None", "Price"]


def test_load_strata_keeps_none_as_a_stratum(monkeypatch) -> None:
    """Non-churners have Churn Category 'None'; that is a real stratum."""
    test_df = pd.DataFrame({"Customer ID": ["a", "b"]})
    eda = pd.DataFrame({
        "Customer ID": ["a", "b"],
        "Churn Category": ["None", "Price"],
    })

    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: eda)
    strata = load_strata(test_df)

    assert "None" in strata


# ---------------------------------------------------------------------------
# _render_e5
# ---------------------------------------------------------------------------
def _client_table():
    return pd.DataFrame([
        {
            "strategy": "fedavg",
            "auc": 0.8889,
            "client_auc_mean": 0.7950,
            "client_auc_std": 0.0045,
            "client_auc_min": 0.7883,
            "client_auc_max": 0.8049,
            "client_auc_gap": 0.0166,
        },
        {
            "strategy": "fedprox",
            "auc": 0.8900,
            "client_auc_mean": 0.7960,
            "client_auc_std": 0.0030,
            "client_auc_min": 0.7900,
            "client_auc_max": 0.8020,
            "client_auc_gap": 0.0120,
        },
    ])


def _reason_table():
    return pd.DataFrame([
        {
            "strategy": "fedavg",
            "reason_recall_mean": 0.3100,
            "reason_recall_std": 0.0500,
            "reason_recall_min": 0.2000,
            "reason_recall_max": 0.4000,
            "reason_recall_gap": 0.2000,
        },
        {
            "strategy": "fedprox",
            "reason_recall_mean": 0.3200,
            "reason_recall_std": 0.0400,
            "reason_recall_min": 0.2400,
            "reason_recall_max": 0.3900,
            "reason_recall_gap": 0.1500,
        },
    ])


def _runs():
    return [
        {
            "strategy": "fedavg",
            "per_reason": [
                {"reason": "Competitor", "n_churned": 170, "n_flagged": 68, "recall": 0.4000},
                {"reason": "Price", "n_churned": 50, "n_flagged": 10, "recall": 0.2000},
            ],
        }
    ]


def test_render_e5_contains_both_sections() -> None:
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 33.3, "fedprox_reason_std_gain_pct": 20.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "E5.1" in md
    assert "E5.2" in md
    assert "label_skew_a0.1" in md


def test_render_e5_reports_pass_when_dispersion_drops() -> None:
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 33.3, "fedprox_reason_std_gain_pct": 20.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "PASS" in md
    assert "33.30" in md


def test_render_e5_reports_fail_when_dispersion_grows() -> None:
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": -10.0, "fedprox_reason_std_gain_pct": -5.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "FAIL" in md


def test_render_e5_always_includes_the_caveat() -> None:
    """The CI caveat is mandatory; it must never be dropped."""
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "±0.13" in md
    assert "40-170 positives" in md


def test_render_e5_explains_why_auc_is_not_used() -> None:
    """The report must state that per-reason AUC is undefined, not just omit it."""
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "100% positive" in md
    assert "AUC" in md


def test_render_e5_shows_per_reason_detail() -> None:
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "Competitor" in md
    assert "Price" in md
    assert "0.4000" in md


def test_render_e5_without_runs_does_not_crash() -> None:
    md = _render_e5(
        _client_table(),
        _reason_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        [],
        "label_skew_a0.1",
    )
    assert "E5.1" in md


def test_min_stratum_samples_is_sane() -> None:
    """Below ~30 churners recall is dominated by a handful of rows."""
    assert MIN_STRATUM_SAMPLES >= 30
