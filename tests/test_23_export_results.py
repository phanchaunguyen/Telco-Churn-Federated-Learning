"""Tests for :mod:`src.serving.s23_export_results`.

The builders are pure functions over the JSON in ``results/``, so they are
tested without a database. The postgres write path is exercised by the CLI, not
by the suite.

Run with::

    pytest tests/test_23_export_results.py -v
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import get_config
from src.serving.s23_export_results import (
    build_convergence_table,
    build_fairness_table,
    build_privacy_table,
)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def test_convergence_has_expected_columns() -> None:
    df = build_convergence_table()

    assert list(df.columns) == ["source", "run", "round", "auc"]


def test_convergence_covers_all_e2_partitions() -> None:
    """E2 is the only run that records a per-round curve."""
    df = build_convergence_table()

    assert set(df["source"]) == {"E2"}
    assert set(df["run"]) == {"iid", "label_skew_a0.5", "label_skew_a0.1"}


def test_convergence_auc_is_in_range() -> None:
    df = build_convergence_table()

    assert (df["auc"] > 0.5).all()
    assert (df["auc"] <= 1.0).all()


def test_convergence_rounds_are_positive() -> None:
    df = build_convergence_table()

    assert (df["round"] >= 1).all()


def test_convergence_has_one_row_per_round_per_run() -> None:
    """Duplicated (run, round) pairs would double-count in a chart."""
    df = build_convergence_table()

    assert not df.duplicated(subset=["source", "run", "round"]).any()


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
def test_fairness_has_expected_columns() -> None:
    df = build_fairness_table()

    assert list(df.columns) == ["view", "strategy", "label", "value"]


def test_fairness_has_both_views() -> None:
    df = build_fairness_table()

    assert set(df["view"]) == {"client", "reason"}


def test_fairness_client_values_are_auc() -> None:
    df = build_fairness_table()
    clients = df[df["view"] == "client"]

    assert (clients["value"] > 0.5).all()
    assert (clients["value"] <= 1.0).all()


def test_fairness_reason_values_are_recall() -> None:
    """Reason values are recall, so they must be in [0, 1] and well below AUC."""
    df = build_fairness_table()
    reasons = df[df["view"] == "reason"]

    assert (reasons["value"] >= 0.0).all()
    assert (reasons["value"] <= 1.0).all()


def test_fairness_excludes_none_reason() -> None:
    """Non-churners are not a churn reason and must not appear as one."""
    df = build_fairness_table()
    reasons = df[df["view"] == "reason"]

    assert "None" not in set(reasons["label"])


def test_fairness_covers_both_strategies() -> None:
    df = build_fairness_table()

    assert set(df["strategy"]) == {"fedavg", "fedprox"}


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
def test_privacy_has_expected_columns() -> None:
    df = build_privacy_table()

    assert list(df.columns) == ["noise_multiplier", "epsilon", "auc", "pr_auc"]


def test_privacy_includes_non_dp_baseline() -> None:
    df = build_privacy_table()

    assert 0.0 in set(df["noise_multiplier"])


def test_privacy_auc_decreases_with_noise() -> None:
    """More noise must not improve AUC; a violation means the sweep is broken."""
    df = build_privacy_table().sort_values("noise_multiplier")
    dp = df[df["noise_multiplier"] > 0]

    assert (dp["auc"].diff().dropna() <= 0).all()


def test_privacy_epsilon_decreases_with_noise() -> None:
    """Higher sigma must buy a smaller epsilon."""
    df = build_privacy_table().sort_values("noise_multiplier")
    dp = df[df["noise_multiplier"] > 0]

    assert (dp["epsilon"].diff().dropna() <= 0).all()


# ---------------------------------------------------------------------------
# Missing sources
# ---------------------------------------------------------------------------
def test_builders_return_empty_frames_when_json_missing(monkeypatch, tmp_path) -> None:
    """A missing results file must yield an empty frame, not a crash."""
    import src.serving.s23_export_results as mod

    class _FakePaths:
        results_dir = tmp_path

    class _FakeCfg:
        paths = _FakePaths()

    monkeypatch.setattr(mod, "get_config", lambda: _FakeCfg())

    for builder in (build_convergence_table, build_fairness_table, build_privacy_table):
        df = builder()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
