"""Tests for :mod:`src.experiments.s16_e2_e3`.

Only the pure helpers are tested here. The full E2/E3 runs take ~25 minutes
each and are exercised by the CLI, not by the test suite.

Run with::

    pytest tests/test_16_e2_e3.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.s16_e2_e3 import (
    CONVERGENCE_SMOOTHING,
    CONVERGENCE_TOLERANCE,
    _render_e2,
    _render_e3,
    _smooth,
    convergence_round,
)


# ---------------------------------------------------------------------------
# _smooth
# ---------------------------------------------------------------------------
def test_smooth_preserves_length() -> None:
    assert len(_smooth([1.0, 2.0, 3.0, 4.0, 5.0], 3)) == 5


def test_smooth_window_one_is_identity() -> None:
    values = [0.1, 0.9, 0.3]
    assert _smooth(values, 1) == values


def test_smooth_averages_centred_window() -> None:
    # window 3, half 1 -> index 1 averages indices 0..2
    out = _smooth([0.0, 3.0, 6.0], 3)
    assert out[1] == pytest.approx(3.0)


def test_smooth_shrinks_window_at_edges() -> None:
    # index 0 only has indices 0..1 available
    out = _smooth([0.0, 4.0, 8.0], 3)
    assert out[0] == pytest.approx(2.0)
    assert out[-1] == pytest.approx(6.0)


def test_smooth_ignores_nan() -> None:
    out = _smooth([1.0, float("nan"), 3.0], 3)
    assert out[1] == pytest.approx(2.0)


def test_smooth_empty() -> None:
    assert _smooth([], 5) == []


# ---------------------------------------------------------------------------
# convergence_round
# ---------------------------------------------------------------------------
def test_convergence_reaches_tolerance() -> None:
    """Final 0.90, tolerance 0.005 -> first round at or above 0.895."""
    aucs = [(1, 0.70), (2, 0.80), (3, 0.896), (4, 0.899), (5, 0.900)]
    assert convergence_round(aucs, window=1) == 3


def test_convergence_never_reaches_returns_total() -> None:
    """A run still improving at the end reports the full length."""
    aucs = [(1, 0.50), (2, 0.60)]
    assert convergence_round(aucs, window=1) == 2


def test_convergence_empty_returns_zero() -> None:
    assert convergence_round([]) == 0


def test_convergence_single_round() -> None:
    assert convergence_round([(1, 0.85)], window=1) == 1


def test_convergence_immediate() -> None:
    """If round 1 already hits the tolerance, that is the answer."""
    aucs = [(1, 0.900), (2, 0.900), (3, 0.900)]
    assert convergence_round(aucs, window=1) == 1


def test_convergence_handles_nan() -> None:
    """NaN rounds are skipped rather than treated as converged."""
    aucs = [(1, float("nan")), (2, 0.80), (3, 0.90)]
    assert convergence_round(aucs, window=1) == 3


def test_convergence_all_nan_returns_total() -> None:
    aucs = [(1, float("nan")), (2, float("nan"))]
    assert convergence_round(aucs, window=1) == 2


def test_convergence_tolerance_is_configurable() -> None:
    aucs = [(1, 0.50), (2, 0.80), (3, 0.90)]
    # tolerance 0.5 -> target 0.40, reached at round 1
    assert convergence_round(aucs, tolerance=0.5, window=1) == 1
    # tolerance 0.001 -> target 0.899, only reached at round 3
    assert convergence_round(aucs, tolerance=0.001, window=1) == 3


def test_convergence_defaults() -> None:
    assert CONVERGENCE_TOLERANCE == pytest.approx(0.005)
    assert CONVERGENCE_SMOOTHING == 5


def test_convergence_flat_curve_does_not_report_round_one() -> None:
    """A flat curve must not be reported as converged at round 1.

    This is the regression that motivated the absolute-tolerance rewrite: with
    a fraction-of-final threshold, a curve that starts at 0.78 and ends at 0.80
    already exceeds 95% of its final value in round 1.
    """
    aucs = [(r, 0.78 + 0.02 * r / 50) for r in range(1, 51)]
    assert convergence_round(aucs) > 1


def test_convergence_uses_peak_not_final_for_diverging_run() -> None:
    """A run that peaks then diverges must not report convergence at round 1.

    FedAvgM with an inverted momentum sign diverged after round 40, dropping
    from 0.789 to 0.764. Measuring against the final value made the opening
    rounds look "converged" because they were already within tolerance of the
    degraded endpoint. The reference must be the peak.
    """
    aucs = [(r, 0.78 + 0.0002 * r) for r in range(1, 41)]
    aucs += [(r, 0.789 - 0.005 * (r - 40)) for r in range(41, 51)]

    assert convergence_round(aucs) > 1


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _e2_table():
    import pandas as pd

    return pd.DataFrame([
        {
            "partition": "iid",
            "auc": 0.8950,
            "auc_ci_lower": 0.8750,
            "auc_ci_upper": 0.9130,
            "pr_auc": 0.7200,
            "recall_at_10pct": 0.33,
            "convergence_round": 12,
            "per_client_auc_std": 0.004,
        },
        {
            "partition": "label_skew_a0.1",
            "auc": 0.8500,
            "auc_ci_lower": 0.8280,
            "auc_ci_upper": 0.8710,
            "pr_auc": 0.6800,
            "recall_at_10pct": 0.30,
            "convergence_round": 38,
            "per_client_auc_std": 0.090,
        },
    ])


def test_render_e2_contains_all_partitions() -> None:
    table = _e2_table()
    gaps = {"iid_gap_pct": 0.0, "label_skew_a0.1_gap_pct": 5.03}
    md = _render_e2(table, gaps)

    assert "iid" in md
    assert "label_skew_a0.1" in md
    assert "0.8950" in md
    assert "0.8500" in md


def test_render_e2_reports_pass_when_skew_hurts() -> None:
    table = _e2_table()
    gaps = {"iid_gap_pct": 0.0, "label_skew_a0.1_gap_pct": 5.03}
    md = _render_e2(table, gaps)
    assert "PASS" in md
    assert "5.03" in md


def test_render_e2_reports_fail_when_skew_helps() -> None:
    table = _e2_table()
    gaps = {"iid_gap_pct": 0.0, "label_skew_a0.1_gap_pct": -1.20}
    md = _render_e2(table, gaps)
    assert "FAIL" in md


def _e3_table():
    import pandas as pd

    return pd.DataFrame([
        {
            "strategy": "fedavg",
            "auc": 0.8500,
            "auc_ci_lower": 0.8280,
            "auc_ci_upper": 0.8710,
            "pr_auc": 0.6800,
            "recall_at_10pct": 0.30,
            "convergence_round": 38,
            "per_client_auc_std": 0.090,
            "per_client_auc_min": 0.6100,
        },
        {
            "strategy": "fedprox",
            "auc": 0.8700,
            "auc_ci_lower": 0.8490,
            "auc_ci_upper": 0.8900,
            "pr_auc": 0.7000,
            "recall_at_10pct": 0.31,
            "convergence_round": 30,
            "per_client_auc_std": 0.060,
            "per_client_auc_min": 0.7000,
        },
    ])


def test_render_e3_contains_all_strategies() -> None:
    table = _e3_table()
    gains = {"fedavg_gain_pct": 0.0, "fedprox_gain_pct": 2.35}
    md = _render_e3(table, gains, "label_skew_a0.1")

    assert "fedavg" in md
    assert "fedprox" in md
    assert "label_skew_a0.1" in md


def test_render_e3_reports_pass_when_alternative_wins() -> None:
    table = _e3_table()
    gains = {"fedavg_gain_pct": 0.0, "fedprox_gain_pct": 2.35}
    md = _render_e3(table, gains, "label_skew_a0.1")
    assert "PASS" in md
    assert "2.35" in md


def test_render_e3_reports_fail_when_fedavg_wins() -> None:
    table = _e3_table()
    gains = {"fedavg_gain_pct": 0.0, "fedprox_gain_pct": -1.50}
    md = _render_e3(table, gains, "label_skew_a0.1")
    assert "FAIL" in md


def test_render_e3_handles_missing_gains() -> None:
    """A table with only fedavg must not crash the renderer."""
    table = _e3_table().head(1)
    md = _render_e3(table, {"fedavg_gain_pct": 0.0}, "label_skew_a0.1")
    assert "fedavg" in md
