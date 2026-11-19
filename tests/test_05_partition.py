"""Tests for :mod:`src.partitioning.s05_partition` and ``verify_silos``.

Run with::

    pytest tests/test_05_partition.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.s01_clean import FEATURE_COLS, ID_COL, TARGET_COL
from src.partitioning.s05_partition import (
    MIN_CLASS_COUNT,
    MIN_CLIENT_SHARE,
    assign_dirichlet,
    assign_iid,
    assign_synthetic_ids,
    validate_partitions,
)


# ---------------------------------------------------------------------------
# Fixtures — a small synthetic frame, no disk access
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5_000
    data = {col: rng.normal(size=n) for col in FEATURE_COLS if col not in (
        "Gender", "Senior Citizen", "Married", "Offer", "Contract",
        "Paperless Billing", "Payment Method", "Phone Service", "Multiple Lines",
        "Internet Service", "Internet Type", "Online Security", "Online Backup",
        "Device Protection Plan", "Premium Tech Support", "Streaming TV",
        "Streaming Movies", "Streaming Music", "Unlimited Data",
    )}
    for col in FEATURE_COLS:
        if col not in data:
            data[col] = rng.choice(["A", "B", "C"], size=n)
    data[TARGET_COL] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Synthetic IDs
# ---------------------------------------------------------------------------
def test_assign_synthetic_ids_is_deterministic(frame: pd.DataFrame) -> None:
    a = assign_synthetic_ids(frame)
    b = assign_synthetic_ids(frame)
    assert list(a[ID_COL]) == list(b[ID_COL])


def test_assign_synthetic_ids_unique(frame: pd.DataFrame) -> None:
    out = assign_synthetic_ids(frame)
    assert out[ID_COL].is_unique
    assert len(out) == len(frame)


def test_assign_synthetic_ids_prefix(frame: pd.DataFrame) -> None:
    out = assign_synthetic_ids(frame)
    assert out[ID_COL].iloc[0].startswith("SYN-")


# ---------------------------------------------------------------------------
# IID
# ---------------------------------------------------------------------------
def test_iid_balances_clients(frame: pd.DataFrame) -> None:
    assignment = assign_iid(frame, k=10, seed=42)
    sizes = np.bincount(assignment, minlength=10)
    assert sizes.max() - sizes.min() <= 1


def test_iid_covers_every_row(frame: pd.DataFrame) -> None:
    assignment = assign_iid(frame, k=10, seed=42)
    assert len(assignment) == len(frame)
    assert (assignment >= 0).all()


def test_iid_preserves_churn_rate(frame: pd.DataFrame) -> None:
    """IID spread must stay within binomial sampling noise for the client size."""
    assignment = assign_iid(frame, k=10, seed=42)
    rates = [frame.loc[assignment == c, TARGET_COL].mean() for c in range(10)]
    pooled = float(frame[TARGET_COL].mean())
    se = float(np.sqrt(pooled * (1 - pooled) / (len(frame) / 10)))
    assert max(rates) - min(rates) < max(4 * se, 0.01)


def test_iid_is_deterministic(frame: pd.DataFrame) -> None:
    a = assign_iid(frame, k=10, seed=42)
    b = assign_iid(frame, k=10, seed=42)
    np.testing.assert_array_equal(a, b)


def test_iid_different_seed_differs(frame: pd.DataFrame) -> None:
    a = assign_iid(frame, k=10, seed=42)
    b = assign_iid(frame, k=10, seed=7)
    assert not np.array_equal(a, b)


def test_iid_rejects_invalid_k(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="k must be"):
        assign_iid(frame, k=0)


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------
def test_dirichlet_covers_every_row(frame: pd.DataFrame) -> None:
    assignment = assign_dirichlet(frame, k=10, alpha=0.5, seed=42)
    assert len(assignment) == len(frame)
    assert (assignment >= 0).all()


def test_dirichlet_uses_all_clients(frame: pd.DataFrame) -> None:
    assignment = assign_dirichlet(frame, k=10, alpha=0.5, seed=42)
    assert set(np.unique(assignment)) == set(range(10))


def test_dirichlet_every_client_has_both_classes(frame: pd.DataFrame) -> None:
    assignment = assign_dirichlet(frame, k=10, alpha=0.1, seed=42)
    for c in range(10):
        part = frame.loc[assignment == c, TARGET_COL]
        assert part.sum() >= MIN_CLASS_COUNT, f"client {c} has too few positives"
        assert (len(part) - part.sum()) >= MIN_CLASS_COUNT, f"client {c} has too few negatives"


def test_dirichlet_severe_alpha_is_more_skewed(frame: pd.DataFrame) -> None:
    """alpha=0.1 must produce a wider churn-rate spread than alpha=0.5."""
    def spread(alpha: float) -> float:
        a = assign_dirichlet(frame, k=10, alpha=alpha, seed=42)
        rates = [frame.loc[a == c, TARGET_COL].mean() for c in range(10)]
        return max(rates) - min(rates)

    assert spread(0.1) > spread(0.5)


def test_dirichlet_is_deterministic(frame: pd.DataFrame) -> None:
    a = assign_dirichlet(frame, k=10, alpha=0.5, seed=42)
    b = assign_dirichlet(frame, k=10, alpha=0.5, seed=42)
    np.testing.assert_array_equal(a, b)


def test_dirichlet_rejects_invalid_alpha(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        assign_dirichlet(frame, k=10, alpha=0.0, seed=42)


def test_dirichlet_respects_min_client_share(frame: pd.DataFrame) -> None:
    assignment = assign_dirichlet(frame, k=10, alpha=0.1, seed=42)
    sizes = np.bincount(assignment, minlength=10)
    min_expected = int(MIN_CLIENT_SHARE * len(frame) / 10)
    assert (sizes >= min_expected).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_partitions_passes_iid(frame: pd.DataFrame) -> None:
    assignment = assign_iid(frame, k=10, seed=42)
    meta = validate_partitions(frame, assignment, 10, strategy="iid")
    assert meta["k"] == 10
    assert meta["n_rows"] == len(frame)
    assert len(meta["sizes"]) == 10


def test_validate_partitions_passes_dirichlet(frame: pd.DataFrame) -> None:
    assignment = assign_dirichlet(frame, k=10, alpha=0.5, seed=42)
    meta = validate_partitions(frame, assignment, 10, strategy="label_skew")
    assert meta["churn_rate_std"] > 0


def test_validate_partitions_rejects_missing_client(frame: pd.DataFrame) -> None:
    assignment = assign_iid(frame, k=10, seed=42)
    bad = assignment.copy()
    bad[bad == 9] = 0
    with pytest.raises(AssertionError, match="Expected clients"):
        validate_partitions(frame, bad, 10, strategy="iid")


def test_validate_partitions_rejects_wrong_length(frame: pd.DataFrame) -> None:
    assignment = assign_iid(frame, k=10, seed=42)[:-1]
    with pytest.raises(AssertionError, match="length mismatch"):
        validate_partitions(frame, assignment, 10, strategy="iid")
