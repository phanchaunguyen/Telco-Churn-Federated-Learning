"""Tests for :mod:`src.experiments.s17_e4_dp`.

The DP-SGD sweep itself takes ~30 minutes, so only the pure helpers and the
trainer's privacy plumbing are tested here.

Run with::

    pytest tests/test_17_e4_dp.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.s17_e4_dp import _render_e4


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _table():
    return pd.DataFrame([
        {
            "noise_multiplier": 0.0,
            "epsilon": 0.0,
            "auc": 0.8950,
            "auc_ci_lower": 0.8750,
            "auc_ci_upper": 0.9130,
            "pr_auc": 0.8120,
            "recall_at_10pct": 0.33,
        },
        {
            "noise_multiplier": 0.5,
            "epsilon": 1.20,
            "auc": 0.8900,
            "auc_ci_lower": 0.8700,
            "auc_ci_upper": 0.9080,
            "pr_auc": 0.8050,
            "recall_at_10pct": 0.32,
        },
        {
            "noise_multiplier": 1.0,
            "epsilon": 2.62,
            "auc": 0.8800,
            "auc_ci_lower": 0.8600,
            "auc_ci_upper": 0.8990,
            "pr_auc": 0.7900,
            "recall_at_10pct": 0.31,
        },
        {
            "noise_multiplier": 2.0,
            "epsilon": 0.72,
            "auc": 0.8600,
            "auc_ci_lower": 0.8390,
            "auc_ci_upper": 0.8800,
            "pr_auc": 0.7600,
            "recall_at_10pct": 0.29,
        },
    ])


def _losses(table: pd.DataFrame, baseline: float) -> dict[str, float]:
    return {
        f"sigma_{r['noise_multiplier']}_loss_pct": (baseline - r["auc"]) / baseline * 100
        for _, r in table.iterrows()
    }


def test_render_e4_contains_every_sigma() -> None:
    table = _table()
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "iid")

    for sigma in ("0.0", "0.5", "1.0", "2.0"):
        assert sigma in md


def test_render_e4_marks_non_dp_epsilon_as_inf() -> None:
    table = _table()
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "iid")
    assert "inf" in md


def test_render_e4_reports_pass_when_monotone() -> None:
    table = _table()
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "iid")
    assert "PASS" in md


def test_render_e4_reports_fail_when_noise_helps() -> None:
    """More noise must never improve AUC; if it does, the sweep is broken."""
    table = _table()
    table.loc[table["noise_multiplier"] == 2.0, "auc"] = 0.9000
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "iid")
    assert "FAIL" in md


def test_render_e4_single_dp_point_is_monotone() -> None:
    """One DP point cannot violate monotonicity."""
    table = _table().head(2)
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "iid")
    assert "PASS" in md


def test_render_e4_includes_partition_name() -> None:
    table = _table()
    md = _render_e4(table, _losses(table, 0.8950), 0.8950, "label_skew_a0.1")
    assert "label_skew_a0.1" in md


# ---------------------------------------------------------------------------
# Trainer privacy plumbing
# ---------------------------------------------------------------------------
def test_trainer_defaults_to_non_private() -> None:
    from src.fl.s09_trainer import LocalTrainer

    X = np.random.randn(200, 8).astype(np.float32)
    y = (np.random.rand(200) > 0.5).astype(np.float32)
    trainer = LocalTrainer(X, y, 8, seed=0)

    assert trainer.noise_multiplier == 0.0
    assert trainer.epsilon == 0.0


def test_trainer_non_private_fit_reports_zero_epsilon() -> None:
    from src.fl.s09_trainer import LocalTrainer

    X = np.random.randn(200, 8).astype(np.float32)
    y = (np.random.rand(200) > 0.5).astype(np.float32)
    trainer = LocalTrainer(X, y, 8, seed=0)
    result = trainer.fit(epochs=1)

    assert trainer.epsilon == 0.0
    assert "epsilon" not in result.extra


def test_trainer_dp_fit_spends_budget() -> None:
    """A private fit must report a finite, positive epsilon."""
    from src.fl.s09_trainer import LocalTrainer

    X = np.random.randn(500, 8).astype(np.float32)
    y = (np.random.rand(500) > 0.5).astype(np.float32)
    trainer = LocalTrainer(X, y, 8, noise_multiplier=1.0, seed=0)
    result = trainer.fit(epochs=1)

    assert np.isfinite(trainer.epsilon)
    assert trainer.epsilon > 0
    assert result.extra["epsilon"] == pytest.approx(trainer.epsilon)


def test_trainer_dp_weights_keep_expected_shapes() -> None:
    """DP wrapping must not change the tensors that cross the wire."""
    from src.fl.s09_trainer import LocalTrainer

    X = np.random.randn(500, 8).astype(np.float32)
    y = (np.random.rand(500) > 0.5).astype(np.float32)

    plain = LocalTrainer(X, y, 8, seed=0).fit(epochs=1)
    private = LocalTrainer(X, y, 8, noise_multiplier=1.0, seed=0).fit(epochs=1)

    assert len(plain.weights) == len(private.weights)
    for a, b in zip(plain.weights, private.weights, strict=True):
        assert a.shape == b.shape
        assert a.dtype == b.dtype


def test_trainer_more_noise_costs_more_privacy_budget() -> None:
    """Higher sigma must yield a smaller epsilon for the same number of steps."""
    from src.fl.s09_trainer import LocalTrainer

    X = np.random.randn(500, 8).astype(np.float32)
    y = (np.random.rand(500) > 0.5).astype(np.float32)

    low = LocalTrainer(X, y, 8, noise_multiplier=0.5, seed=0)
    low.fit(epochs=1)
    high = LocalTrainer(X, y, 8, noise_multiplier=2.0, seed=0)
    high.fit(epochs=1)

    assert high.epsilon < low.epsilon
