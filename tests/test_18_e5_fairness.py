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
    with pytest.raises(KeyError, match="no Churn Category"):
        load_strata(test_df)


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


def _stratum_table():
    return pd.DataFrame([
        {
            "strategy": "fedavg",
            "stratum_auc_std": 0.0500,
            "stratum_auc_min": 0.7000,
            "stratum_auc_gap": 0.1500,
        },
        {
            "strategy": "fedprox",
            "stratum_auc_std": 0.0400,
            "stratum_auc_min": 0.7200,
            "stratum_auc_gap": 0.1200,
        },
    ])


def _runs():
    return [
        {
            "strategy": "fedavg",
            "per_stratum": [
                {"stratum": "None", "n_rows": 944, "n_churned": 0, "auc": float("nan")},
                {"stratum": "Competitor", "n_rows": 170, "n_churned": 170, "auc": 0.7100},
            ],
        }
    ]


def test_render_e5_contains_both_sections() -> None:
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": 33.3, "fedprox_stratum_std_gain_pct": 20.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "E5.1" in md
    assert "E5.2" in md
    assert "label_skew_a0.1" in md


def test_render_e5_reports_pass_when_dispersion_drops() -> None:
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": 33.3, "fedprox_stratum_std_gain_pct": 20.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "PASS" in md
    assert "33.30" in md


def test_render_e5_reports_fail_when_dispersion_grows() -> None:
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": -10.0, "fedprox_stratum_std_gain_pct": -5.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "FAIL" in md


def test_render_e5_always_includes_the_caveat() -> None:
    """The +/-0.10 CI caveat is mandatory; it must never be dropped."""
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "±0.10" in md
    assert "75 churned rows" in md


def test_render_e5_renders_nan_stratum_as_na() -> None:
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        _runs(),
        "label_skew_a0.1",
    )
    assert "n/a" in md


def test_render_e5_without_runs_does_not_crash() -> None:
    md = _render_e5(
        _client_table(),
        _stratum_table(),
        {"fedprox_client_std_gain_pct": 1.0},
        [],
        "label_skew_a0.1",
    )
    assert "E5.1" in md


def test_min_stratum_samples_is_sane() -> None:
    """Below ~30 rows AUC is dominated by a handful of positives."""
    assert MIN_STRATUM_SAMPLES >= 30
