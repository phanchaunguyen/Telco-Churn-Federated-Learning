"""Synthetic data amplification via a rank-based Gaussian Copula.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

The generator is fitted on ``train_real`` ONLY. ``test_real`` is never seen by
the model — that is what makes the TSTR evaluation meaningful.

Why a hand-rolled Copula instead of ``copulas.GaussianMultivariate``
--------------------------------------------------------------------
The library fits a Gaussian KDE per column and inverts it with a root-finding
routine that evaluates the kernel against every training row. Sampling 1M rows
therefore needs ``n_rows x n_train`` float64 cells (~39 GiB here) and is
prohibitively slow.

The standard alternative — implemented here — is the **rank-based Gaussian
Copula**:

1. Map each column to uniform scores via its empirical CDF.
2. Map uniforms to normal scores via the inverse normal CDF.
3. Estimate the correlation matrix of the normal scores.
4. Draw ``N(0, Sigma)`` samples.
5. Map back to uniforms, then to values via the inverse empirical CDF.

This is O(n·d) in time and memory, preserves marginals exactly (we invert the
empirical CDF), and preserves the rank-correlation structure.

Artifacts produced
------------------
``synthetic_1m.parquet``  1,000,000 rows x 30 cols (29 features + churn).
``amplify_metadata.json`` encoders, quantile grids, achieved churn rate.

Usage
-----
>>> from src.data.amplify import build_synthetic
>>> df = build_synthetic(n_rows=1_000_000, write=True)
>>> df.shape
(1000000, 30)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_config
from src.data.clean import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, TARGET_COL

logger = logging.getLogger(__name__)

# Columns that must be integers after sampling
INT_LIKE_COLS: list[str] = [
    "Age",
    "Number of Dependents",
    "Tenure in Months",
    "Number of Referrals",
    "Avg Monthly GB Download",
]

# Tolerance on the achieved churn rate
MAX_CHURN_RATE_DRIFT: float = 0.01

# Number of quantile grid points stored per column for the inverse CDF
N_QUANTILE_GRID: int = 10_000


# =============================================================================
# Encoding
# =============================================================================
def fit_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build a category -> integer map for every categorical column."""
    encoders: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        values = sorted(df[col].astype(str).unique())
        encoders[col] = {v: i for i, v in enumerate(values)}
    return encoders


def encode(df: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Apply the category maps, producing an all-numeric frame."""
    out = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col in encoders:
            mapping = encoders[col]
            out[col] = df[col].astype(str).map(mapping).fillna(0).astype(float)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    out[TARGET_COL] = df[TARGET_COL].astype(float)
    return out


# =============================================================================
# Gaussian Copula — fit
# =============================================================================
def _to_normal_scores(values: np.ndarray) -> np.ndarray:
    """Map raw values to normal scores via the empirical CDF.

    Ties are broken by average rank, which keeps the transform monotone and
    avoids the ``±inf`` a naive ``rank / n`` would produce at the tails.
    """
    n = len(values)
    ranks = stats.rankdata(values, method="average")
    # (rank - 0.5) / n keeps the scores strictly inside (0, 1)
    uniforms = (ranks - 0.5) / n
    return stats.norm.ppf(uniforms)


def fit_copula(df: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    """Fit a rank-based Gaussian Copula on ``train_real``.

    Returns
    -------
    dict
        ``correlation``    — (d x d) correlation matrix of normal scores
        ``quantiles``      — per-column quantile grid for the inverse CDF
        ``encoders``       — category maps
        ``numeric_ranges`` — observed [min, max] per numeric column
        plus bookkeeping fields.
    """
    encoders = fit_encoders(df)
    encoded = encode(df, encoders)
    columns = list(encoded.columns)

    logger.info("Fitting Gaussian Copula on %s rows x %s cols", *encoded.shape)

    # 1. Normal scores per column
    normal_scores = np.empty_like(encoded.to_numpy(dtype=float))
    for j, col in enumerate(columns):
        normal_scores[:, j] = _to_normal_scores(encoded[col].to_numpy(dtype=float))

    # 2. Correlation of normal scores (the copula parameter)
    correlation = np.corrcoef(normal_scores, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    # 3. Quantile grid per column for the inverse empirical CDF
    grid = np.linspace(0.0, 1.0, N_QUANTILE_GRID)
    quantiles: dict[str, list[float]] = {}
    for col in columns:
        quantiles[col] = np.quantile(encoded[col].to_numpy(dtype=float), grid).tolist()

    numeric_ranges = {
        col: (float(df[col].min()), float(df[col].max())) for col in NUMERIC_COLS
    }

    return {
        "columns": columns,
        "correlation": correlation,
        "quantiles": quantiles,
        "encoders": encoders,
        "numeric_ranges": numeric_ranges,
        "n_train_rows": int(len(df)),
        "train_churn_rate": float(df[TARGET_COL].mean()),
        "seed": seed,
    }


# =============================================================================
# Gaussian Copula — sample
# =============================================================================
def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone.

    ``np.corrcoef`` on finite data is PSD in theory, but floating-point error
    can push the smallest eigenvalue slightly negative, which makes
    ``np.random.multivariate_normal`` raise. Clipping eigenvalues fixes it.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-8, None)
    projected = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(projected))
    return projected / np.outer(d, d)


def sample_copula(
    state: dict[str, Any],
    n_rows: int,
    *,
    seed: int = 42,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Draw ``n_rows`` synthetic rows from a fitted copula state."""
    rng = np.random.default_rng(seed)
    columns = state["columns"]
    correlation = _nearest_psd(state["correlation"])
    quantiles = state["quantiles"]
    grid = np.linspace(0.0, 1.0, N_QUANTILE_GRID)

    logger.info("Sampling %s synthetic rows in chunks of %s", f"{n_rows:,}", f"{chunk_size:,}")

    frames: list[pd.DataFrame] = []
    remaining = n_rows
    while remaining > 0:
        size = min(chunk_size, remaining)

        # 1. Correlated normal draws
        normal = rng.multivariate_normal(
            mean=np.zeros(len(columns)),
            cov=correlation,
            size=size,
            method="cholesky",
        )
        # 2. Back to uniforms
        uniforms = stats.norm.cdf(normal)

        # 3. Inverse empirical CDF per column
        out = {col: np.interp(uniforms[:, j], grid, quantiles[col])
               for j, col in enumerate(columns)}
        frames.append(pd.DataFrame(out))

        remaining -= size
        logger.info("  sampled %s / %s", f"{n_rows - remaining:,}", f"{n_rows:,}")

    raw = pd.concat(frames, ignore_index=True)
    return decode(raw, state["encoders"], state["numeric_ranges"])


# =============================================================================
# Decoding
# =============================================================================
def decode(
    df: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
    numeric_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Invert the encoding and repair out-of-range numeric values."""
    out = pd.DataFrame(index=df.index)

    for col in FEATURE_COLS:
        if col in encoders:
            inverse = {i: v for v, i in encoders[col].items()}
            codes = df[col].round().astype(int).clip(0, len(inverse) - 1)
            out[col] = codes.map(inverse)
        else:
            values = df[col].astype(float)
            lo, hi = numeric_ranges[col]
            values = values.clip(lo, hi)
            if col in INT_LIKE_COLS:
                values = values.round()
            out[col] = values

    out[TARGET_COL] = (df[TARGET_COL] >= 0.5).astype(int)
    return out[FEATURE_COLS + [TARGET_COL]]


# =============================================================================
# Validation
# =============================================================================
def validate_synthetic(df: pd.DataFrame, *, expected_churn_rate: float) -> None:
    """Assert the synthetic frame is structurally sound.

    The churn-rate tolerance scales with the sample size. A fixed 1% band is
    right for 1M rows but rejects a perfectly valid 10K draw, where the
    binomial standard error alone is ~0.45%. The tolerance is therefore
    ``max(MAX_CHURN_RATE_DRIFT, 4 * binomial SE)``.
    """
    assert list(df.columns) == FEATURE_COLS + [TARGET_COL], (
        f"Unexpected column order: {list(df.columns)}"
    )
    assert len(df) > 0, "Empty synthetic frame"
    assert not df.isna().any().any(), "Synthetic frame contains NaN"
    assert set(df[TARGET_COL].unique()) == {0, 1}, "Target must be binary"

    n = len(df)
    se = float(np.sqrt(expected_churn_rate * (1 - expected_churn_rate) / n))
    tolerance = max(MAX_CHURN_RATE_DRIFT, 4 * se)

    drift = abs(float(df[TARGET_COL].mean()) - expected_churn_rate)
    assert drift < tolerance, (
        f"churn rate drift {drift:.4f} exceeds tolerance {tolerance:.4f} "
        f"(n={n:,}, binomial SE {se:.4f})"
    )

    for col in CATEGORICAL_COLS:
        assert df[col].dtype == object, f"{col} should be categorical, got {df[col].dtype}"

    logger.info(
        "validate_synthetic: OK (%s rows, churn drift %.4f < %.4f)",
        f"{n:,}",
        drift,
        tolerance,
    )


# =============================================================================
# Orchestration
# =============================================================================
def build_synthetic(
    train_path: str | Path | None = None,
    *,
    n_rows: int | None = None,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fit on ``train_real``, sample ``n_rows``, validate, and optionally write."""
    cfg = get_config()
    src_path = Path(train_path) if train_path is not None else cfg.paths.train_real_parquet
    if not src_path.exists():
        raise FileNotFoundError(
            f"train_real not found: {src_path}. Run `python -m src.data.split` first."
        )

    train = pd.read_parquet(src_path)
    logger.info("Loaded %s (%s rows x %s cols)", src_path, *train.shape)

    n = n_rows if n_rows is not None else cfg.amplification.n_rows
    state = fit_copula(train, seed=cfg.split.seed)
    synthetic = sample_copula(state, n, seed=cfg.split.seed)

    validate_synthetic(synthetic, expected_churn_rate=state["train_churn_rate"])

    if write:
        target_dir = Path(out_dir) if out_dir is not None else cfg.paths.synthetic_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        out_path = target_dir / cfg.paths.synthetic_parquet.name
        synthetic.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%s rows x %s cols)", out_path, *synthetic.shape)

        meta_path = target_dir / "amplify_metadata.json"
        meta = {
            "method": "rank_gaussian_copula",
            "n_rows": int(len(synthetic)),
            "n_train_rows": state["n_train_rows"],
            "train_churn_rate": state["train_churn_rate"],
            "synthetic_churn_rate": float(synthetic[TARGET_COL].mean()),
            "seed": state["seed"],
            "numeric_ranges": {k: list(v) for k, v in state["numeric_ranges"].items()},
            "encoders": state["encoders"],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote %s", meta_path)

    return synthetic


def main() -> None:
    """CLI entry point: ``python -m src.data.amplify``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    df = build_synthetic(write=True)

    print("\n" + "=" * 70)
    print("AMPLIFY STEP COMPLETE")
    print("=" * 70)
    print(f"  synthetic_1m.parquet : {df.shape[0]:>9,} rows x {df.shape[1]:>2} cols")
    print(f"  churn rate           : {df[TARGET_COL].mean():.4f}")
    print(f"  churned              : {int(df[TARGET_COL].sum()):,}")
    print(f"  NaN cells            : {int(df.isna().sum().sum())}")
    print("=" * 70)


if __name__ == "__main__":
    main()
