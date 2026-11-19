"""Data cleaning for the IBM Telco EXTENDED churn dataset.

Pipeline position: **CLEAN → AMPLIFY → PARTITION**

This module is the single source of truth for which columns are features and
which are dropped (and why). Downstream modules (amplification, partitioning,
feature engineering) MUST import ``FEATURE_COLS`` fr>om here rather than
re-declaring their own lists.

Artifacts produced
------------------
``eda_7043.parquet``   (D0.5) 50 columns, ``Joined`` rows removed, values untouched.
``clean_7043.parquet`` (D0.6) 31 columns = ``Customer ID`` + 29 features + ``churn``.

Usage
-----
>>> from src.data.s01_clean import build_clean
>>> df_clean, df_eda = build_clean(write=True)
>>> df_clean.shape
(..., 31)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)

# =============================================================================
# Column taxonomy — single source of truth
# =============================================================================

ID_COL: str = "Customer ID"
TARGET_COL: str = "churn"
STATUS_COL: str = "Customer Status"
LABEL_COL: str = "Churn Label"
DROP_STATUS_VALUE: str = "Joined"

# Known header defects in the source CSV: the raw header contains
# ``"Total Refu    nds"`` (four spaces injected mid-word). Normalisation maps
# the corrupted spelling back to the canonical name.
HEADER_FIXES: dict[str, str] = {
    "Total Refu    nds": "Total Refunds",
}

# --- Loại 1: TARGET LEAKAGE (direct) ----------------------------------------
TARGET_LEAKAGE_COLS: list[str] = [LABEL_COL, STATUS_COL]

# --- Loại 2: POST-HOC DERIVED ----------------------------------------------
POSTHOC_DERIVED_COLS: list[str] = ["Churn Score", "CLTV", "Satisfaction Score"]

# --- Loại 3: INDIRECT POST-HOC ---------------------------------------------
INDIRECT_POSTHOC_COLS: list[str] = [
    "Total Revenue",
    "Total Refunds",
    "Churn Category",
    "Churn Reason",
]

# --- Loại 4: ZERO-VARIANCE / CONSTANTS -------------------------------------
CONSTANT_COLS: list[str] = ["Country", "State", "Quarter"]

# --- Loại 5: HIGH-CARDINALITY GEOGRAPHIC (EDA only) ------------------------
GEO_COLS: list[str] = ["City", "Zip Code", "Latitude", "Longitude", "Population"]

# --- Loại 6: DERIVED DUPLICATES --------------------------------------------
DERIVED_DUP_COLS: list[str] = ["Under 30", "Referred a Friend", "Dependents"]

# --- All columns that must never reach a client silo -----------------------
LEAKAGE_COLS: list[str] = (
    TARGET_LEAKAGE_COLS + POSTHOC_DERIVED_COLS + INDIRECT_POSTHOC_COLS
)

# --- Columns dropped from features but preserved in the EDA artifact -------
EDA_ONLY_COLS: list[str] = (
    POSTHOC_DERIVED_COLS + INDIRECT_POSTHOC_COLS + GEO_COLS + CONSTANT_COLS
)

# =============================================================================
# The 29 features (5 demographics + 6 account + 12 services + 6 usage/billing)
# =============================================================================
DEMOGRAPHIC_COLS: list[str] = [
    "Gender",
    "Age",
    "Senior Citizen",
    "Married",
    "Number of Dependents",
]

ACCOUNT_COLS: list[str] = [
    "Tenure in Months",
    "Offer",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Number of Referrals",
]

SERVICE_COLS: list[str] = [
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]

USAGE_BILLING_COLS: list[str] = [
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
]

FEATURE_COLS: list[str] = (
    DEMOGRAPHIC_COLS + ACCOUNT_COLS + SERVICE_COLS + USAGE_BILLING_COLS
)

# --- Columns that are categorical (needed later for one-hot encoding) ------
CATEGORICAL_COLS: list[str] = [
    "Gender",
    "Senior Citizen",
    "Married",
    "Offer",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
]

NUMERIC_COLS: list[str] = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

# --- Every column that is intentionally removed from the feature matrix ----
DROPPED_COLS: list[str] = (
    TARGET_LEAKAGE_COLS
    + POSTHOC_DERIVED_COLS
    + INDIRECT_POSTHOC_COLS
    + CONSTANT_COLS
    + GEO_COLS
    + DERIVED_DUP_COLS
)

EXPECTED_RAW_COLUMNS: int = 50
EXPECTED_FEATURE_COUNT: int = 29
EXPECTED_CLEAN_COLUMNS: int = 31  # ID + 29 features + target


# =============================================================================
# Loading
# =============================================================================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Repair known header defects in the source CSV.

    The raw file spells one column ``"Total Refu    nds"`` (four spaces injected
    mid-word). Without this fix the column would silently escape the leakage
    drop list and leak into the feature matrix.
    """
    renames = {bad: good for bad, good in HEADER_FIXES.items() if bad in df.columns}
    if renames:
        logger.warning("Repaired malformed header(s): %s", renames)
        df = df.rename(columns=renames)

    # Defensive: collapse any remaining runs of whitespace inside header names
    collapsed = {c: re.sub(r"\s+", " ", c).strip() for c in df.columns}
    if collapsed != dict(zip(df.columns, df.columns, strict=True)):
        df = df.rename(columns=collapsed)
    return df


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw IBM Telco EXTENDED CSV.

    Parameters
    ----------
    path:
        Optional override. Defaults to ``cfg.paths.raw_data``.

    Returns
    -------
    pandas.DataFrame
        The raw frame, 7043 x 50, with headers normalised.
    """
    cfg = get_config()
    csv_path = Path(path) if path is not None else cfg.paths.raw_data
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {csv_path}")

    # ``keep_default_na=False`` is essential: the dataset uses the literal
    # string "None" as a valid category for ``Offer`` and ``Internet Type``.
    # Pandas' default NA list includes "None", which would silently turn those
    # categories into NaN. Only truly empty fields become NaN.
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    df = normalize_headers(df)
    logger.info("Loaded raw dataset: %s rows x %s cols", *df.shape)

    if df.shape[1] != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_RAW_COLUMNS} raw columns, got {df.shape[1]}. "
            "This is not the IBM Telco EXTENDED dataset."
        )
    return df


# =============================================================================
# Cleaning steps
# =============================================================================
def drop_joined(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where ``Customer Status == 'Joined'``.

    Newly joined customers have not had a full observation window, so their
    churn label is not comparable to established customers. They are dropped
    BEFORE amplification so synthetic data never learns from them.
    """
    if STATUS_COL not in df.columns:
        raise KeyError(f"Missing required column: {STATUS_COL!r}")

    n_before = len(df)
    out = df[df[STATUS_COL] != DROP_STATUS_VALUE].copy()
    n_dropped = n_before - len(out)
    logger.info("Dropped %s 'Joined' rows (%s -> %s)", n_dropped, n_before, len(out))
    return out


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary ``churn`` target from ``Customer Status``.

    ``Churned`` -> 1, everything else (``Stayed``) -> 0.
    ``Churn Label`` is cross-checked for consistency and then dropped.
    """
    if STATUS_COL not in df.columns:
        raise KeyError(f"Missing required column: {STATUS_COL!r}")

    out = df.copy()
    out[TARGET_COL] = (out[STATUS_COL] == "Churned").astype("int8")

    # Cross-check against the redundant Churn Label column when present
    if LABEL_COL in out.columns:
        mismatch = (out[LABEL_COL] == "Yes").astype("int8") != out[TARGET_COL]
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            raise ValueError(
                f"{n_mismatch} rows where Churn Label disagrees with Customer Status"
            )
        logger.info("Churn Label agrees with Customer Status on all %s rows", len(out))

    return out


def drop_columns(df: pd.DataFrame, columns: list[str], *, reason: str) -> pd.DataFrame:
    """Drop ``columns`` that are present, logging how many were removed."""
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("Columns not found (skipped): %s", missing)
    logger.info("Dropping %s columns (%s): %s", len(present), reason, present)
    return df.drop(columns=present)


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Loại 1 + 2 + 3 leakage columns."""
    return drop_columns(df, LEAKAGE_COLS, reason="leakage Loại 1-3")


def drop_constants(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Loại 4 zero-variance columns."""
    return drop_columns(df, CONSTANT_COLS, reason="zero-variance Loại 4")


def drop_derived_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Loại 6 derived-duplicate columns (keep the more granular variant)."""
    return drop_columns(df, DERIVED_DUP_COLS, reason="derived duplicates Loại 6")


def drop_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Loại 5 high-cardinality geographic columns."""
    return drop_columns(df, GEO_COLS, reason="geo high-cardinality Loại 5")


# =============================================================================
# Assertions
# =============================================================================
def validate_clean(df: pd.DataFrame) -> None:
    """Assert that a cleaned frame is safe to amplify and partition.

    Raises
    ------
    AssertionError
        If any invariant is violated.
    """
    # 1. No leakage column survived (checked first: most informative failure)
    leaked = [c for c in DROPPED_COLS if c in df.columns]
    assert not leaked, f"Leakage/dropped columns still present: {leaked}"

    # 2. Shape
    assert df.shape[1] == EXPECTED_CLEAN_COLUMNS, (
        f"Expected {EXPECTED_CLEAN_COLUMNS} columns, got {df.shape[1]}: {list(df.columns)}"
    )

    # 3. Feature count
    assert len(FEATURE_COLS) == EXPECTED_FEATURE_COUNT, (
        f"FEATURE_COLS has {len(FEATURE_COLS)} entries, expected {EXPECTED_FEATURE_COUNT}"
    )

    # 4. Every declared feature is present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"Declared features missing from frame: {missing}"

    # 5. Target is binary and complete
    assert TARGET_COL in df.columns, f"Missing target column {TARGET_COL!r}"
    assert set(df[TARGET_COL].unique()) <= {0, 1}, "Target must be binary {0, 1}"
    assert df[TARGET_COL].notna().all(), "Target contains NaN"

    # 6. No NaN in features
    nan_cols = [c for c in FEATURE_COLS if df[c].isna().any()]
    assert not nan_cols, f"Features contain NaN: {nan_cols}"

    # 7. No 'Joined' rows remain
    if STATUS_COL in df.columns:
        assert DROP_STATUS_VALUE not in set(df[STATUS_COL].unique()), (
            "'Joined' rows still present"
        )

    # 8. Customer ID is unique
    assert df[ID_COL].is_unique, f"{ID_COL} is not unique"

    logger.info("validate_clean: all invariants hold (%s rows)", len(df))


def validate_column_accounting(raw_columns: list[str]) -> None:
    """Assert that features + dropped + ID exactly cover the raw 50 columns."""
    covered = set(FEATURE_COLS) | set(DROPPED_COLS) | {ID_COL}
    raw = set(raw_columns)

    unaccounted = raw - covered
    unknown = covered - raw
    assert not unaccounted, f"Raw columns not classified: {sorted(unaccounted)}"
    assert not unknown, f"Declared columns absent from raw data: {sorted(unknown)}"
    assert len(covered) == EXPECTED_RAW_COLUMNS, (
        f"Accounting covers {len(covered)} columns, expected {EXPECTED_RAW_COLUMNS}"
    )
    logger.info("validate_column_accounting: all %s raw columns classified", len(covered))


# =============================================================================
# Orchestration
# =============================================================================
def build_clean(
    raw_path: str | Path | None = None,
    *,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full CLEAN step.

    Parameters
    ----------
    raw_path:
        Optional override for the raw CSV location.
    write:
        When ``True`` (default), write both parquet artifacts to disk.
    out_dir:
        Optional override for the output directory.

    Returns
    -------
    (df_clean, df_eda)
        ``df_clean`` — 31 columns, ready for amplification.
        ``df_eda``   — 50 columns minus ``Joined`` rows, for EDA only.
    """
    cfg = get_config()
    raw = load_raw(raw_path)

    # Column accounting must hold on the raw schema before anything is dropped
    validate_column_accounting(list(raw.columns))

    # --- D0.5: EDA artifact (50 cols, Joined removed, values untouched) -----
    df_eda = drop_joined(raw)

    # --- D0.6: clean artifact ----------------------------------------------
    df = derive_target(df_eda)
    df = drop_leakage(df)
    df = drop_constants(df)
    df = drop_derived_duplicates(df)
    df = drop_geo(df)

    # Keep ID + features + target, in a deterministic order
    df_clean = df[[ID_COL, *FEATURE_COLS, TARGET_COL]].copy()

    validate_clean(df_clean)

    if write:
        target_dir = Path(out_dir) if out_dir is not None else cfg.paths.processed_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        eda_path = target_dir / cfg.paths.eda_parquet.name
        clean_path = target_dir / cfg.paths.clean_parquet.name

        df_eda.to_parquet(eda_path, index=False)
        df_clean.to_parquet(clean_path, index=False)

        logger.info("Wrote %s (%s rows x %s cols)", eda_path, *df_eda.shape)
        logger.info("Wrote %s (%s rows x %s cols)", clean_path, *df_clean.shape)

    return df_clean, df_eda


def main() -> None:
    """CLI entry point: ``python -m src.data.s01_clean``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    df_clean, df_eda = build_clean(write=True)

    print("\n" + "=" * 70)
    print("CLEAN STEP COMPLETE")
    print("=" * 70)
    print(f"  eda_7043.parquet   : {df_eda.shape[0]:>6} rows x {df_eda.shape[1]:>2} cols")
    print(f"  clean_7043.parquet : {df_clean.shape[0]:>6} rows x {df_clean.shape[1]:>2} cols")
    print(f"  features           : {len(FEATURE_COLS)}")
    print(f"  churn rate         : {df_clean[TARGET_COL].mean():.4f}")
    print(f"  churn counts       : {df_clean[TARGET_COL].value_counts().to_dict()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
