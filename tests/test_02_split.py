"""Tests for :mod:`src.data.s02_split`.

Run with::

    pytest tests/test_02_split.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.s01_clean import (
    EXPECTED_CLEAN_COLUMNS,
    FEATURE_COLS,
    ID_COL,
    TARGET_COL,
    build_clean,
)
from src.data.s02_split import (
    MAX_CHURN_RATE_DELTA,
    build_split,
    split_real,
    validate_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    df, _ = build_clean(write=False)
    return df


@pytest.fixture(scope="module")
def split(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_real(clean, test_size=0.2, seed=42)


# ---------------------------------------------------------------------------
# Shape and schema
# ---------------------------------------------------------------------------
def test_split_shapes(split: tuple[pd.DataFrame, pd.DataFrame], clean: pd.DataFrame) -> None:
    train, test = split
    assert len(train) + len(test) == len(clean)
    assert len(test) / len(clean) == pytest.approx(0.2, abs=0.01)


def test_split_preserves_columns(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    assert train.shape[1] == EXPECTED_CLEAN_COLUMNS
    assert test.shape[1] == EXPECTED_CLEAN_COLUMNS
    assert list(train.columns) == list(test.columns)


def test_split_preserves_feature_order(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = split
    assert list(train.columns) == [ID_COL, *FEATURE_COLS, TARGET_COL]


# ---------------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------------
def test_split_is_disjoint(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    assert not (set(train[ID_COL]) & set(test[ID_COL]))


def test_split_covers_all_customers(
    split: tuple[pd.DataFrame, pd.DataFrame], clean: pd.DataFrame
) -> None:
    train, test = split
    assert set(train[ID_COL]) | set(test[ID_COL]) == set(clean[ID_COL])


def test_split_ids_unique_within_each_half(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    assert train[ID_COL].is_unique
    assert test[ID_COL].is_unique


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def test_split_preserves_churn_rate(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    delta = abs(train[TARGET_COL].mean() - test[TARGET_COL].mean())
    assert delta < MAX_CHURN_RATE_DELTA


def test_split_has_both_classes(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    for part in (train, test):
        assert set(part[TARGET_COL].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_split_is_deterministic(clean: pd.DataFrame) -> None:
    a_train, a_test = split_real(clean, test_size=0.2, seed=42)
    b_train, b_test = split_real(clean, test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_different_seed_gives_different_split(clean: pd.DataFrame) -> None:
    a_train, _ = split_real(clean, test_size=0.2, seed=42)
    b_train, _ = split_real(clean, test_size=0.2, seed=7)
    assert set(a_train[ID_COL]) != set(b_train[ID_COL])


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
def test_validate_split_passes(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    validate_split(train, test, expected_test_size=0.2)


def test_validate_split_rejects_overlap(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    bad_test = test.copy()
    bad_test.loc[bad_test.index[0], ID_COL] = train[ID_COL].iloc[0]
    with pytest.raises(AssertionError, match="both splits"):
        validate_split(train, bad_test, expected_test_size=0.2)


def test_validate_split_rejects_wrong_fraction(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = split
    with pytest.raises(AssertionError, match="test fraction"):
        validate_split(train, test, expected_test_size=0.5)


def test_split_rejects_invalid_test_size(clean: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="test_size"):
        split_real(clean, test_size=1.5)


def test_split_rejects_missing_stratify_column(clean: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not in frame"):
        split_real(clean, stratify_on="does_not_exist")


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def test_build_split_without_write() -> None:
    train, test = build_split(write=False)
    assert len(train) > 0 and len(test) > 0
    assert train.shape[1] == EXPECTED_CLEAN_COLUMNS
    assert test.shape[1] == EXPECTED_CLEAN_COLUMNS
