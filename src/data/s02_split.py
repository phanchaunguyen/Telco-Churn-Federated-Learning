"""Stratified splitting of the clean dataset into train_real / test_real.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

Why split BEFORE amplification
-------------------------------
If the Copula were fitted on the full clean cohort, it would learn the marginal
distributions and correlations of the rows that later become the test set. The
synthetic data would then "know" the test distribution, and evaluating on it
would be circular — it would measure in-distribution fit, not generalisation.

Splitting first guarantees that ``test_real`` is never seen by the generator.

Artifacts produced
------------------
``train_real.parquet``  80% of clean_7043, the ONLY input to amplification.
``test_real.parquet``   20% of clean_7043, the PRIMARY evaluation set.
                        NEVER amplified, NEVER partitioned.

Usage
-----
>>> from src.data.s02_split import build_split
>>> train, test = build_split(write=True)
>>> train.shape, test.shape
((5271, 31), (1318, 31))
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.data.s01_clean import ID_COL, TARGET_COL, validate_clean

logger = logging.getLogger(__name__)

# Tolerances for the stratified-split invariants
MAX_CHURN_RATE_DELTA: float = 0.01
EXPECTED_TRAIN_FRACTION: float = 0.8


# =============================================================================
# Core split
# =============================================================================
def split_real(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_on: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a clean frame into ``(train_real, test_real)``.

    Parameters
    ----------
    df:
        Clean frame with ``Customer ID``, 29 features and ``churn``.
    test_size:
        Fraction held out for evaluation (default 0.2).
    seed:
        Random seed for reproducibility.
    stratify_on:
        Column used to preserve the class ratio in both halves.

    Returns
    -------
    (train, test)
        Both frames keep the original column order and index reset.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if stratify_on not in df.columns:
        raise KeyError(f"Stratify column {stratify_on!r} not in frame")

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_on],
        shuffle=True,
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(
        "Split %s rows -> train %s / test %s (test_size=%.2f, seed=%s)",
        len(df),
        len(train),
        len(test),
        test_size,
        seed,
    )
    return train, test


# =============================================================================
# Validation
# =============================================================================
def validate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    expected_test_size: float = 0.2,
) -> None:
    """Assert the invariants that make ``test_real`` a valid hold-out set.

    Raises
    ------
    AssertionError
        If any invariant is violated.
    """
    # 1. Both halves are individually valid clean frames
    validate_clean(train)
    validate_clean(test)

    # 2. No customer appears in both halves
    overlap = set(train[ID_COL]) & set(test[ID_COL])
    assert not overlap, f"{len(overlap)} Customer IDs appear in both splits"

    # 3. Every customer is accounted for exactly once
    assert len(train) + len(test) > 0, "Empty split"
    assert train[ID_COL].is_unique and test[ID_COL].is_unique, "Duplicate IDs within a split"

    # 4. Test fraction matches the request
    actual_test_size = len(test) / (len(train) + len(test))
    assert abs(actual_test_size - expected_test_size) < 0.01, (
        f"test fraction {actual_test_size:.4f} deviates from {expected_test_size}"
    )

    # 5. Stratification preserved the churn rate
    train_rate = float(train[TARGET_COL].mean())
    test_rate = float(test[TARGET_COL].mean())
    delta = abs(train_rate - test_rate)
    assert delta < MAX_CHURN_RATE_DELTA, (
        f"churn rate drift {delta:.4f} exceeds {MAX_CHURN_RATE_DELTA} "
        f"(train {train_rate:.4f} vs test {test_rate:.4f})"
    )

    # 6. Both classes present in both halves
    for name, part in (("train", train), ("test", test)):
        assert set(part[TARGET_COL].unique()) == {0, 1}, (
            f"{name} split does not contain both classes"
        )

    logger.info(
        "validate_split: OK (train %s rows, churn %.4f | test %s rows, churn %.4f)",
        len(train),
        train_rate,
        len(test),
        test_rate,
    )


# =============================================================================
# Orchestration
# =============================================================================
def build_split(
    clean_path: str | Path | None = None,
    *,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``clean_7043.parquet``, split it, validate, and optionally write.

    Parameters
    ----------
    clean_path:
        Optional override for the clean parquet location.
    write:
        When ``True`` (default), write ``train_real`` and ``test_real``.
    out_dir:
        Optional override for the output directory.

    Returns
    -------
    (train_real, test_real)
    """
    cfg = get_config()
    src_path = Path(clean_path) if clean_path is not None else cfg.paths.clean_parquet
    if not src_path.exists():
        raise FileNotFoundError(
            f"Clean artifact not found: {src_path}. Run `bash scripts/02_run_clean.sh` first."
        )

    df = pd.read_parquet(src_path)
    logger.info("Loaded %s (%s rows x %s cols)", src_path, *df.shape)

    train, test = split_real(
        df,
        test_size=cfg.split.test_size,
        seed=cfg.split.seed,
        stratify_on=cfg.split.stratify_on,
    )
    validate_split(train, test, expected_test_size=cfg.split.test_size)

    if write:
        target_dir = Path(out_dir) if out_dir is not None else cfg.paths.processed_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        train_path = target_dir / cfg.paths.train_real_parquet.name
        test_path = target_dir / cfg.paths.test_real_parquet.name

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        logger.info("Wrote %s (%s rows x %s cols)", train_path, *train.shape)
        logger.info("Wrote %s (%s rows x %s cols)", test_path, *test.shape)

    return train, test


def main() -> None:
    """CLI entry point: ``python -m src.data.s02_split``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    train, test = build_split(write=True)

    print("\n" + "=" * 70)
    print("SPLIT STEP COMPLETE")
    print("=" * 70)
    print(f"  train_real.parquet : {train.shape[0]:>6} rows x {train.shape[1]:>2} cols")
    print(f"  test_real.parquet  : {test.shape[0]:>6} rows x {test.shape[1]:>2} cols")
    print(f"  train churn rate   : {train[TARGET_COL].mean():.4f}")
    print(f"  test churn rate    : {test[TARGET_COL].mean():.4f}")
    print(f"  train churned      : {int(train[TARGET_COL].sum()):,}")
    print(f"  test churned       : {int(test[TARGET_COL].sum()):,}")
    print("=" * 70)
    print("  test_real is the PRIMARY evaluation set.")
    print("  It must NEVER be amplified or partitioned.")
    print("=" * 70)


# =============================================================================
# Synthetic split (90/10)
# =============================================================================
def split_synthetic(
    df: pd.DataFrame,
    *,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the synthetic frame into ``(synth_train, synth_test)``.

    ``synth_train`` is the only input to partitioning. ``synth_test`` exists
    solely for scale experiments (E6) and as a sanity check (E7); it is never
    reported as a main result.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET_COL],
        shuffle=True,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(
        "Split synthetic %s rows -> synth_train %s / synth_test %s",
        f"{len(df):,}",
        f"{len(train):,}",
        f"{len(test):,}",
    )
    return train, test


def build_synthetic_split(
    synthetic_path: str | Path | None = None,
    *,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``synthetic_1m.parquet``, split 90/10, and optionally write."""
    cfg = get_config()
    src_path = (
        Path(synthetic_path) if synthetic_path is not None else cfg.paths.synthetic_parquet
    )
    if not src_path.exists():
        raise FileNotFoundError(
            f"Synthetic artifact not found: {src_path}. Run `python -m src.data.s03_amplify` first."
        )

    df = pd.read_parquet(src_path)
    logger.info("Loaded %s (%s rows x %s cols)", src_path, *df.shape)

    train, test = split_synthetic(
        df,
        test_size=cfg.split.synthetic_test_size,
        seed=cfg.split.seed,
    )

    # Invariants
    assert len(train) + len(test) == len(df), "Synthetic split lost rows"
    assert not train.isna().any().any(), "synth_train contains NaN"
    assert not test.isna().any().any(), "synth_test contains NaN"
    assert set(train[TARGET_COL].unique()) == {0, 1}, "synth_train missing a class"
    assert set(test[TARGET_COL].unique()) == {0, 1}, "synth_test missing a class"

    if write:
        target_dir = Path(out_dir) if out_dir is not None else cfg.paths.synthetic_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        train_path = target_dir / cfg.paths.synth_train_parquet.name
        test_path = target_dir / cfg.paths.synth_test_parquet.name

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        logger.info("Wrote %s (%s rows x %s cols)", train_path, *train.shape)
        logger.info("Wrote %s (%s rows x %s cols)", test_path, *test.shape)

    return train, test


def main_synthetic() -> None:
    """CLI entry point: ``python -m src.data.s02_split --synthetic``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    train, test = build_synthetic_split(write=True)

    print("\n" + "=" * 70)
    print("SYNTHETIC SPLIT COMPLETE")
    print("=" * 70)
    print(f"  synth_train.parquet : {train.shape[0]:>9,} rows x {train.shape[1]:>2} cols")
    print(f"  synth_test.parquet  : {test.shape[0]:>9,} rows x {test.shape[1]:>2} cols")
    print(f"  synth_train churn   : {train[TARGET_COL].mean():.4f}")
    print(f"  synth_test churn    : {test[TARGET_COL].mean():.4f}")
    print("=" * 70)
    print("  synth_train -> partitioning")
    print("  synth_test  -> E6/E7 scale experiments ONLY")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if "--synthetic" in sys.argv:
        main_synthetic()
    else:
        main()
