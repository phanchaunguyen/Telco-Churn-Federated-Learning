"""Tests for :mod:`src.data.s01_clean`.

Run with::

    pytest tests/test_01_clean.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.s01_clean import (
    ACCOUNT_COLS,
    CATEGORICAL_COLS,
    CONSTANT_COLS,
    DEMOGRAPHIC_COLS,
    DERIVED_DUP_COLS,
    DROPPED_COLS,
    EXPECTED_CLEAN_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_RAW_COLUMNS,
    FEATURE_COLS,
    GEO_COLS,
    ID_COL,
    INDIRECT_POSTHOC_COLS,
    LEAKAGE_COLS,
    NUMERIC_COLS,
    POSTHOC_DERIVED_COLS,
    SERVICE_COLS,
    TARGET_COL,
    TARGET_LEAKAGE_COLS,
    USAGE_BILLING_COLS,
    build_clean,
    derive_target,
    drop_joined,
    load_raw,
    validate_clean,
    validate_column_accounting,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_clean(write=False)


# ---------------------------------------------------------------------------
# Column taxonomy
# ---------------------------------------------------------------------------
def test_feature_count_is_29() -> None:
    assert len(FEATURE_COLS) == EXPECTED_FEATURE_COUNT == 29


def test_feature_groups_sum_to_29() -> None:
    assert len(DEMOGRAPHIC_COLS) == 5
    assert len(ACCOUNT_COLS) == 6
    assert len(SERVICE_COLS) == 12
    assert len(USAGE_BILLING_COLS) == 6
    assert (
        len(DEMOGRAPHIC_COLS)
        + len(ACCOUNT_COLS)
        + len(SERVICE_COLS)
        + len(USAGE_BILLING_COLS)
        == 29
    )


def test_feature_cols_have_no_duplicates() -> None:
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_categorical_and_numeric_partition_features() -> None:
    assert set(CATEGORICAL_COLS) | set(NUMERIC_COLS) == set(FEATURE_COLS)
    assert not (set(CATEGORICAL_COLS) & set(NUMERIC_COLS))


def test_number_of_referrals_is_a_feature() -> None:
    """Regression guard: this column was missing from the original 28-item list."""
    assert "Number of Referrals" in FEATURE_COLS
    assert "Referred a Friend" not in FEATURE_COLS


def test_leakage_groups() -> None:
    assert TARGET_LEAKAGE_COLS == ["Churn Label", "Customer Status"]
    assert POSTHOC_DERIVED_COLS == ["Churn Score", "CLTV", "Satisfaction Score"]
    assert INDIRECT_POSTHOC_COLS == [
        "Total Revenue",
        "Total Refunds",
        "Churn Category",
        "Churn Reason",
    ]
    assert len(LEAKAGE_COLS) == 9


def test_dropped_cols_cover_all_six_categories() -> None:
    assert len(CONSTANT_COLS) == 3
    assert len(GEO_COLS) == 5
    assert len(DERIVED_DUP_COLS) == 3
    assert len(DROPPED_COLS) == 9 + 3 + 5 + 3


def test_no_feature_is_also_dropped() -> None:
    assert not (set(FEATURE_COLS) & set(DROPPED_COLS))


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
def test_raw_shape(raw: pd.DataFrame) -> None:
    assert raw.shape == (7043, EXPECTED_RAW_COLUMNS)


def test_raw_has_no_churn_value_column(raw: pd.DataFrame) -> None:
    """The EXTENDED dataset has no 1/0 Churn Value column."""
    assert "Churn Value" not in raw.columns
    assert "Churn Score Category" not in raw.columns
    assert "Satisfaction Score Label" not in raw.columns
    assert "Count" not in raw.columns


def test_total_charges_is_float(raw: pd.DataFrame) -> None:
    assert pd.api.types.is_float_dtype(raw["Total Charges"])


def test_customer_status_has_three_values(raw: pd.DataFrame) -> None:
    assert set(raw["Customer Status"].unique()) == {"Churned", "Stayed", "Joined"}


def test_column_accounting_covers_all_50(raw: pd.DataFrame) -> None:
    validate_column_accounting(list(raw.columns))


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def test_drop_joined_removes_only_joined(raw: pd.DataFrame) -> None:
    out = drop_joined(raw)
    assert "Joined" not in set(out["Customer Status"].unique())
    assert len(out) < len(raw)
    assert set(out["Customer Status"].unique()) == {"Churned", "Stayed"}


def test_derive_target_is_binary(raw: pd.DataFrame) -> None:
    out = derive_target(drop_joined(raw))
    assert set(out[TARGET_COL].unique()) == {0, 1}
    assert out[TARGET_COL].dtype == "int8"


def test_derive_target_matches_churn_label(raw: pd.DataFrame) -> None:
    out = derive_target(drop_joined(raw))
    expected = (out["Churn Label"] == "Yes").astype("int8")
    pd.testing.assert_series_equal(
        out[TARGET_COL], expected, check_names=False, check_dtype=False
    )


# ---------------------------------------------------------------------------
# Clean artifact
# ---------------------------------------------------------------------------
def test_clean_shape(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    assert df_clean.shape[1] == EXPECTED_CLEAN_COLUMNS == 31


def test_clean_column_order(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    assert list(df_clean.columns) == [ID_COL, *FEATURE_COLS, TARGET_COL]


def test_clean_has_no_leakage(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    for col in DROPPED_COLS:
        assert col not in df_clean.columns, f"leakage column survived: {col}"


def test_clean_has_no_nan(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    assert not df_clean.isna().any().any()


def test_clean_target_balance(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    rate = df_clean[TARGET_COL].mean()
    assert 0.20 < rate < 0.35, f"churn rate {rate:.3f} outside expected range"


def test_clean_ids_unique(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    assert df_clean[ID_COL].is_unique


def test_validate_clean_passes(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    validate_clean(df_clean)


def test_validate_clean_rejects_leakage(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    bad = df_clean.copy()
    bad["Churn Score"] = 50
    with pytest.raises(AssertionError, match="Leakage"):
        validate_clean(bad)


def test_validate_clean_rejects_nan(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, _ = built
    bad = df_clean.copy()
    bad.loc[bad.index[0], "Age"] = None
    with pytest.raises(AssertionError, match="NaN"):
        validate_clean(bad)


# ---------------------------------------------------------------------------
# EDA artifact
# ---------------------------------------------------------------------------
def test_eda_keeps_all_50_columns(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, df_eda = built
    assert df_eda.shape[1] == EXPECTED_RAW_COLUMNS


def test_eda_keeps_leakage_for_analysis(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, df_eda = built
    for col in ["Churn Score", "CLTV", "Satisfaction Score", "Churn Category", "Churn Reason"]:
        assert col in df_eda.columns


def test_eda_and_clean_have_same_rows(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_clean, df_eda = built
    assert len(df_clean) == len(df_eda)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_build_clean_is_deterministic() -> None:
    a, _ = build_clean(write=False)
    b, _ = build_clean(write=False)
    pd.testing.assert_frame_equal(a, b)
