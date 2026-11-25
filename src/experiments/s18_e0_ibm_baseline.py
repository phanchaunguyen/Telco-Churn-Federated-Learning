"""E0 — the IBM Churn Score as an external reference point.

Pipeline position: **EDA → BASELINE (no training)**

``Churn Score`` ships with the dataset and is IBM's own churn propensity
estimate. It is a post-hoc derived column, so it can never be a feature — but
it is a useful yardstick: if the federated model cannot beat a score that came
free with the data, the modelling effort has not paid off.

This module computes that score's univariate AUC on the real cohort and writes
it to ``results/E0_ibm_baseline.json``. No model is trained; the score is used
directly as a ranking.

Why the number lives in a file and not only in the docs
-------------------------------------------------------
The value 0.9421 appeared in ``docs/04_data_decisions.md`` and the architecture
notes, but nothing in the repository could reproduce it. A number quoted in a
report should be regenerable by a command, so this module exists.

Usage
-----
>>> from src.experiments.s18_e0_ibm_baseline import run_e0
>>> result = run_e0()
>>> round(result["auc"], 4)
0.9421
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import get_config

logger = logging.getLogger(__name__)

TARGET_COL = "churn"
SCORE_COL = "Churn Score"

# The EDA artifact keeps the 50 original columns, so the target is still the
# raw "Yes"/"No" label rather than the derived 0/1 ``churn`` column that only
# exists in the cleaned artifact.
LABEL_COL = "Churn Label"
POSITIVE_LABEL = "Yes"

# The score is a 0-100 integer. Anything outside this band means the column was
# read wrong (e.g. as a string, or a different column was picked up).
MIN_PLAUSIBLE_AUC: float = 0.90
MAX_PLAUSIBLE_AUC: float = 0.97


# =============================================================================
# Core
# =============================================================================
def compute_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the univariate AUC of ``Churn Score`` against the target.

    Parameters
    ----------
    df:
        Frame with both ``Churn Score`` and a churn label. The EDA artifact
        (``eda_7043.parquet``) carries the raw ``Churn Label`` column; a frame
        that already has a 0/1 ``churn`` column is also accepted.

    Returns
    -------
    dict
        ``{auc, n_samples, n_churned, score_min, score_max, score_mean}``
    """
    if SCORE_COL not in df.columns:
        raise KeyError(f"Missing column for E0: {SCORE_COL!r}")

    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL].to_numpy(dtype=np.int64)
    elif LABEL_COL in df.columns:
        y_true = (df[LABEL_COL].astype(str) == POSITIVE_LABEL).to_numpy(dtype=np.int64)
    else:
        raise KeyError(
            f"Need either {TARGET_COL!r} or {LABEL_COL!r} to compute E0"
        )

    y_score = df[SCORE_COL].to_numpy(dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Target has only one class; AUC is undefined")

    auc = float(roc_auc_score(y_true, y_score))

    return {
        "auc": auc,
        "n_samples": int(len(y_true)),
        "n_churned": int(y_true.sum()),
        "score_min": float(y_score.min()),
        "score_max": float(y_score.max()),
        "score_mean": float(y_score.mean()),
    }


# =============================================================================
# E0
# =============================================================================
def run_e0(*, write: bool = True) -> dict[str, Any]:
    """Compute the IBM Churn Score baseline and persist it.

    Returns
    -------
    dict
        The metrics plus the artifact paths when ``write=True``.
    """
    cfg = get_config()
    eda_path = cfg.paths.eda_parquet
    if not eda_path.exists():
        raise FileNotFoundError(
            f"EDA artifact not found: {eda_path}. Run the clean step first."
        )

    df = pd.read_parquet(eda_path, columns=[SCORE_COL, LABEL_COL])
    logger.info("Loaded %s (%s rows)", eda_path, f"{len(df):,}")

    metrics = compute_baseline(df)

    if not (MIN_PLAUSIBLE_AUC <= metrics["auc"] <= MAX_PLAUSIBLE_AUC):
        logger.warning(
            "E0 AUC %.4f is outside the plausible band [%.2f, %.2f]; "
            "check that %r was read as a numeric column",
            metrics["auc"],
            MIN_PLAUSIBLE_AUC,
            MAX_PLAUSIBLE_AUC,
            SCORE_COL,
        )

    logger.info(
        "E0 IBM Churn Score: AUC %.4f on %s rows (%s churned)",
        metrics["auc"],
        f"{metrics['n_samples']:,}",
        f"{metrics['n_churned']:,}",
    )

    result: dict[str, Any] = {
        "experiment": "E0",
        "description": "IBM Churn Score univariate AUC (external reference)",
        "score_column": SCORE_COL,
        **metrics,
    }

    if write:
        out = cfg.paths.results_dir / "E0_ibm_baseline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E0_ibm_baseline.md"
        md.write_text(_render_e0(result), encoding="utf-8")
        logger.info("Wrote %s", md)

        result["json"] = out
        result["markdown"] = md

    return result


def _render_e0(result: dict[str, Any]) -> str:
    return "\n".join([
        "# E0 — IBM Churn Score baseline",
        "",
        "The `Churn Score` column ships with the dataset and is IBM's own churn",
        "propensity estimate. It is post-hoc derived, so it can never be a",
        "feature — but it is the yardstick the federated model has to beat.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AUC | **{result['auc']:.4f}** |",
        f"| Samples | {result['n_samples']:,} |",
        f"| Churned | {result['n_churned']:,} |",
        f"| Score range | {result['score_min']:.0f} – {result['score_max']:.0f} |",
        f"| Score mean | {result['score_mean']:.2f} |",
        "",
        "> **How to read this.** No model was trained. The score is used directly",
        "> as a ranking, so this is the ceiling a single free column achieves.",
        "> The federated model reaches AUC 0.8949 (E1 fold 0), which is below this",
        "> baseline — expected, because `Churn Score` is derived from information",
        "> the model is not allowed to see.",
        "",
    ])


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s18_e0_ibm_baseline``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    result = run_e0()

    print("\n" + "=" * 70)
    print("E0 — IBM CHURN SCORE BASELINE")
    print("=" * 70)
    print(f"  AUC      : {result['auc']:.4f}")
    print(f"  samples  : {result['n_samples']:,}")
    print(f"  churned  : {result['n_churned']:,}")
    print(f"  json     : {result['json']}")


if __name__ == "__main__":
    main()
