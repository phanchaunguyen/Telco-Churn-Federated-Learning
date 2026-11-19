"""Synthetic data quality gate.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

Three independent checks decide whether the synthetic data is usable:

1. **KS test** per column — does each marginal match the real one?
2. **Correlation diff** — is the joint structure preserved?
3. **TSTR** (Train-on-Synthetic, Test-on-Real) — does a model trained purely on
   synthetic data generalise to ``test_real``?

TSTR is the decisive check. KS and correlation can both look fine while the
synthetic data is useless for learning, because they only inspect marginals and
pairwise structure. TSTR measures what actually matters.

Reference: Jordon et al. (2022), "Synthetic Data — what, why and how?"

Artifacts produced
------------------
``quality_report.json``  per-column KS results, correlation diff, TSTR gap.

Usage
-----
>>> from src.data.s04_quality_gate import run_quality_gate
>>> report = run_quality_gate(write=True)
>>> report["passed"]
True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config import get_config
from src.data.s01_clean import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, TARGET_COL

logger = logging.getLogger(__name__)

# Pass thresholds (see docs/13_layer1_plan.md section 4, Phase 4)
MIN_KS_PASS_RATE: float = 0.90
MAX_CORRELATION_DIFF: float = 0.10
MAX_TSTR_AUC_GAP: float = 0.05


# =============================================================================
# Check 1 — KS test per column
# =============================================================================
def ks_test_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Two-sample KS test on every numeric column.

    Categorical columns are compared by total-variation distance instead, since
    KS is only defined for continuous distributions.
    """
    results: dict[str, Any] = {}

    for col in NUMERIC_COLS:
        stat, p_value = stats.ks_2samp(
            real[col].to_numpy(dtype=float),
            synthetic[col].to_numpy(dtype=float),
        )
        results[col] = {
            "test": "ks",
            "statistic": float(stat),
            "p_value": float(p_value),
            "passed": bool(p_value >= 0.05),
        }

    for col in CATEGORICAL_COLS:
        real_dist = real[col].astype(str).value_counts(normalize=True)
        synth_dist = synthetic[col].astype(str).value_counts(normalize=True)
        categories = real_dist.index.union(synth_dist.index)
        tvd = 0.5 * float(
            (real_dist.reindex(categories, fill_value=0)
             - synth_dist.reindex(categories, fill_value=0)).abs().sum()
        )
        results[col] = {
            "test": "total_variation",
            "statistic": tvd,
            "p_value": None,
            "passed": bool(tvd < 0.10),
        }

    return results


# =============================================================================
# Check 2 — correlation structure
# =============================================================================
def correlation_diff(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Compare the Spearman correlation structure of real vs synthetic.

    The raw Frobenius norm grows with the number of columns (it is a sum of
    squared entries), so a fixed threshold on it is not comparable across
    feature sets. Two scale-invariant summaries are reported instead:

    * ``mean_abs_diff`` — average absolute difference over the off-diagonal
      entries. This is the headline number and the one the gate uses.
    * ``max_abs_diff`` — worst single pair, useful for spotting one broken
      relationship that the average would hide.
    """
    real_corr = real[NUMERIC_COLS].corr(method="spearman").fillna(0.0).to_numpy()
    synth_corr = synthetic[NUMERIC_COLS].corr(method="spearman").fillna(0.0).to_numpy()

    diff = np.abs(real_corr - synth_corr)
    # Ignore the diagonal (always 1.0 on both sides)
    off_diagonal = diff[~np.eye(diff.shape[0], dtype=bool)]

    mean_abs = float(off_diagonal.mean())
    max_abs = float(off_diagonal.max())
    frobenius = float(np.linalg.norm(real_corr - synth_corr, ord="fro"))

    return {
        "mean_abs_diff": mean_abs,
        "max_abs_diff": max_abs,
        "frobenius_diff": frobenius,
        "n_pairs": int(off_diagonal.size),
        "passed": bool(mean_abs < MAX_CORRELATION_DIFF),
    }


# =============================================================================
# Check 3 — TSTR
# =============================================================================
def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals so a sklearn model can consume the frame."""
    return pd.get_dummies(df[FEATURE_COLS], columns=CATEGORICAL_COLS, drop_first=False)


def tstr_auc(
    synthetic: pd.DataFrame,
    test_real: pd.DataFrame,
    *,
    train_real: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train on synthetic, test on real. Returns the AUC and the gap vs real-trained.

    The reference model is trained on ``train_real`` and evaluated on
    ``test_real`` — the same held-out set the synthetic model is scored on.
    That makes the gap a like-for-like comparison of *training data quality*,
    not of in-sample fit.
    """
    x_synth = _encode_for_model(synthetic)
    x_test = _encode_for_model(test_real)

    # Align columns — synthetic may lack a rare category present in test_real
    x_test = x_test.reindex(columns=x_synth.columns, fill_value=0)

    y_synth = synthetic[TARGET_COL].to_numpy()
    y_test = test_real[TARGET_COL].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=seed,
    )
    model.fit(x_synth, y_synth)
    proba = model.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    # Reference: same architecture trained on REAL data, scored on the same
    # held-out test_real. This is the honest upper bound for this feature set.
    if train_real is not None:
        x_train = _encode_for_model(train_real).reindex(columns=x_synth.columns, fill_value=0)
        ref_model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
        )
        ref_model.fit(x_train, train_real[TARGET_COL].to_numpy())
        ref_auc = float(roc_auc_score(y_test, ref_model.predict_proba(x_test)[:, 1]))
        ref_source = "train_real -> test_real"
    else:
        ref_auc = float("nan")
        ref_source = "unavailable"

    gap = abs(ref_auc - auc) if np.isfinite(ref_auc) else float("nan")
    return {
        "tstr_auc": auc,
        "reference_auc": ref_auc,
        "reference_source": ref_source,
        "auc_gap": gap,
        "passed": bool(np.isfinite(gap) and gap < MAX_TSTR_AUC_GAP),
    }


# =============================================================================
# Orchestration
# =============================================================================
def run_quality_gate(
    synthetic_path: str | Path | None = None,
    test_real_path: str | Path | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Run all three checks and return a consolidated report."""
    cfg = get_config()
    synth_path = (
        Path(synthetic_path) if synthetic_path is not None else cfg.paths.synthetic_parquet
    )
    test_path = (
        Path(test_real_path) if test_real_path is not None else cfg.paths.test_real_parquet
    )

    for path in (synth_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    synthetic = pd.read_parquet(synth_path)
    test_real = pd.read_parquet(test_path)
    logger.info("Loaded synthetic %s and test_real %s", synthetic.shape, test_real.shape)

    # The real reference for marginal/correlation checks is train_real, because
    # that is what the generator was fitted on. test_real is reserved for TSTR.
    train_real = pd.read_parquet(cfg.paths.train_real_parquet)

    ks = ks_test_columns(train_real, synthetic)
    ks_pass_rate = sum(1 for r in ks.values() if r["passed"]) / len(ks)

    corr = correlation_diff(train_real, synthetic)
    tstr = tstr_auc(synthetic, test_real, train_real=train_real, seed=cfg.split.seed)

    passed = (
        ks_pass_rate >= MIN_KS_PASS_RATE
        and corr["passed"]
        and tstr["passed"]
    )

    report: dict[str, Any] = {
        "passed": bool(passed),
        "thresholds": {
            "min_ks_pass_rate": MIN_KS_PASS_RATE,
            "max_correlation_diff": MAX_CORRELATION_DIFF,
            "max_tstr_auc_gap": MAX_TSTR_AUC_GAP,
        },
        "ks": {
            "pass_rate": ks_pass_rate,
            "n_columns": len(ks),
            "n_passed": sum(1 for r in ks.values() if r["passed"]),
            "per_column": ks,
        },
        "correlation": corr,
        "tstr": tstr,
        "shapes": {
            "train_real": list(train_real.shape),
            "synthetic": list(synthetic.shape),
            "test_real": list(test_real.shape),
        },
    }

    if write:
        out_path = cfg.paths.quality_report
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    return report


def main() -> None:
    """CLI entry point: ``python -m src.data.s04_quality_gate``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    report = run_quality_gate(write=True)

    print("\n" + "=" * 70)
    print("SYNTHETIC QUALITY GATE")
    print("=" * 70)
    print(f"  KS pass rate        : {report['ks']['pass_rate']:.1%} "
          f"({report['ks']['n_passed']}/{report['ks']['n_columns']}) "
          f"[threshold {MIN_KS_PASS_RATE:.0%}]")
    print(f"  Correlation mean|d|  : {report['correlation']['mean_abs_diff']:.4f} "
          f"[threshold {MAX_CORRELATION_DIFF}]")
    print(f"  Correlation max|d|   : {report['correlation']['max_abs_diff']:.4f}")
    print(f"  TSTR AUC            : {report['tstr']['tstr_auc']:.4f}")
    print(f"  Reference AUC       : {report['tstr']['reference_auc']:.4f} "
          f"({report['tstr']['reference_source']})")
    print(f"  TSTR gap            : {report['tstr']['auc_gap']:.4f} "
          f"[threshold {MAX_TSTR_AUC_GAP}]")
    print("-" * 70)
    print(f"  RESULT              : {'PASS' if report['passed'] else 'FAIL'}")
    print("=" * 70)

    if not report["passed"]:
        failed = [c for c, r in report["ks"]["per_column"].items() if not r["passed"]]
        if failed:
            print(f"  Columns failing KS/TVD: {failed}")


if __name__ == "__main__":
    main()
