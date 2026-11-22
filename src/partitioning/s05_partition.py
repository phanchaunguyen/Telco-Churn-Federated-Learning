"""Client partitioning: IID and Dirichlet label skew.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

Two strategies only (see docs/12_architecture.md section 2):

* **IID** — shuffle and split evenly. Every client sees roughly the global
  churn rate. This is the baseline that isolates the effect of non-IID-ness.
* **Label skew** — draw per-client class proportions from ``Dirichlet(alpha)``.
  ``alpha = 0.5`` is moderate skew, ``alpha = 0.1`` is severe. Small alpha
  pushes clients toward single-class silos, which is what FedProx and FedAvgM
  are designed to survive.

Synthetic rows carry no real ``Customer ID``, so a deterministic synthetic ID
(``SYN-0000001`` ...) is generated here. It exists purely so the silo
verification step can assert disjointness.

Artifacts produced
------------------
``results/partition_metadata.json``  per-client size, churn rate, strategy.

Usage
-----
>>> from src.partitioning.s05_partition import build_partitions
>>> parts = build_partitions(strategy="label_skew", alpha=0.1, k=10)
>>> len(parts)
10
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.s01_clean import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

Strategy = Literal["iid", "label_skew"]

# A Dirichlet draw can hand a client almost nothing, leaving a silo too small to
# train on. Every client is guaranteed at least this share of each class before
# the proportions are renormalised.
MIN_CLIENT_SHARE: float = 0.02

# Every client must see at least this many rows of each class, otherwise the
# local BCE loss is undefined and the model cannot learn a decision boundary.
MIN_CLASS_COUNT: int = 20

SYNTHETIC_ID_PREFIX: str = "SYN-"


# =============================================================================
# Synthetic identifiers
# =============================================================================
def assign_synthetic_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic ``Customer ID`` to a synthetic frame."""
    out = df.copy()
    out.insert(0, "Customer ID", [f"{SYNTHETIC_ID_PREFIX}{i:07d}" for i in range(len(out))])
    return out


# =============================================================================
# IID
# =============================================================================
def assign_iid(df: pd.DataFrame, k: int, *, seed: int = 42) -> np.ndarray:
    """Return a client index per row, balanced and shuffled."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    assignment = np.empty(len(df), dtype=int)
    assignment[order] = np.arange(len(df)) % k
    return assignment


# =============================================================================
# Dirichlet label skew
# =============================================================================
def assign_dirichlet(
    df: pd.DataFrame,
    k: int,
    alpha: float,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return a client index per row, with class proportions drawn from Dir(alpha).

    For each class, the share of that class handed to each client is drawn from
    a Dirichlet distribution. Small ``alpha`` concentrates a class onto a few
    clients, which is exactly the non-IID regime the experiments target.

    A raw Dirichlet draw can leave a client with a handful of rows, which is
    untrainable. The standard remedy (used by FedML, Flower tutorials, and the
    original Niid-Bench code) is to floor the proportions at
    ``MIN_CLIENT_SHARE`` and renormalise. This preserves the skew — a client
    that drew 0.001 still ends up far below one that drew 0.6 — while
    guaranteeing every silo has usable data.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    rng = np.random.default_rng(seed)
    assignment = np.full(len(df), -1, dtype=int)
    labels = df[TARGET_COL].to_numpy()

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_cls = len(idx)

        proportions = rng.dirichlet(np.repeat(alpha, k))
        # Floor then renormalise — keeps the relative skew, removes the zeros
        proportions = np.maximum(proportions, MIN_CLIENT_SHARE)
        proportions = proportions / proportions.sum()

        counts = np.floor(proportions * n_cls).astype(int)
        # Guarantee every client sees at least MIN_CLASS_COUNT rows of this class
        counts = np.maximum(counts, min(MIN_CLASS_COUNT, n_cls // k))
        # Distribute the rounding remainder to the largest fractional parts
        remainder = n_cls - counts.sum()
        if remainder > 0:
            fractional = proportions * n_cls - counts
            order = np.argsort(-fractional)
            counts[order[:remainder]] += 1
        elif remainder < 0:
            # The floor pushed the total over; trim from the largest clients
            while counts.sum() > n_cls:
                counts[int(np.argmax(counts))] -= 1

        start = 0
        for client, count in enumerate(counts):
            assignment[idx[start:start + count]] = client
            start += count

    if (assignment < 0).any():
        raise RuntimeError("Some rows were not assigned to a client")
    return assignment


# =============================================================================
# Validation
# =============================================================================
def validate_partitions(
    df: pd.DataFrame,
    assignment: np.ndarray,
    k: int,
    *,
    strategy: str,
) -> dict[str, Any]:
    """Assert the partition is complete, disjoint, and structurally sound."""
    assert len(assignment) == len(df), "Assignment length mismatch"
    assert set(np.unique(assignment)) == set(range(k)), (
        f"Expected clients 0..{k - 1}, got {sorted(np.unique(assignment))}"
    )

    sizes = np.bincount(assignment, minlength=k)
    assert sizes.sum() == len(df), "Partition does not cover every row"
    min_expected = int(MIN_CLIENT_SHARE * len(df) / k)
    assert (sizes >= min_expected).all(), (
        f"Client below {min_expected} samples: {sizes.tolist()}"
    )

    churn_rates = [
        float(df.loc[assignment == c, TARGET_COL].mean()) for c in range(k)
    ]

    # Every client must contain both classes, otherwise BCE loss is undefined
    # and the local model cannot learn a decision boundary at all.
    for c in range(k):
        n_pos = int(df.loc[assignment == c, TARGET_COL].sum())
        n_neg = int(sizes[c] - n_pos)
        assert n_pos >= MIN_CLASS_COUNT, (
            f"Client {c} has only {n_pos} churned rows (need >= {MIN_CLASS_COUNT})"
        )
        assert n_neg >= MIN_CLASS_COUNT, (
            f"Client {c} has only {n_neg} retained rows (need >= {MIN_CLASS_COUNT})"
        )

    if strategy == "iid":
        # The spread across clients is pure sampling noise, so the tolerance
        # must scale with both the per-client sample size and the number of
        # clients. Two things make a fixed threshold wrong:
        #
        #   1. Smaller silos have larger binomial standard error.
        #   2. The max-minus-min of k samples grows with k even when every
        #      sample is drawn from the same distribution. For k=10 the
        #      expected range is about 3.1 SE and the 99.9th percentile is
        #      about 4.5 SE, so a 4 SE bound rejects a correct partitioner
        #      roughly one run in a hundred.
        #
        # The bound below is 5 SE with a 0.02 floor. At 90k rows per client
        # that is ~13 SE, which no correct partitioner would ever exceed, so
        # the check still catches real bugs (a client receiving a wildly
        # different churn rate) while tolerating normal variation.
        pooled = float(df[TARGET_COL].mean())
        se = float(np.sqrt(pooled * (1 - pooled) / max(sizes.min(), 1)))
        tolerance = max(5 * se, 0.02)
        spread = max(churn_rates) - min(churn_rates)
        assert spread < tolerance, (
            f"IID churn-rate spread {spread:.4f} exceeds tolerance {tolerance:.4f} "
            f"(pooled {pooled:.4f}, min client size {int(sizes.min())}, "
            f"binomial SE {se:.4f})"
        )

    metadata = {
        "strategy": strategy,
        "k": k,
        "n_rows": int(len(df)),
        "sizes": sizes.tolist(),
        "churn_rates": [round(r, 6) for r in churn_rates],
        "churn_rate_min": round(min(churn_rates), 6),
        "churn_rate_max": round(max(churn_rates), 6),
        "churn_rate_std": round(float(np.std(churn_rates)), 6),
    }
    logger.info(
        "validate_partitions: OK (%s, k=%s, sizes %s..%s, churn %.3f..%.3f)",
        strategy,
        k,
        int(sizes.min()),
        int(sizes.max()),
        min(churn_rates),
        max(churn_rates),
    )
    return metadata


# =============================================================================
# Orchestration
# =============================================================================
def build_partitions(
    strategy: Strategy,
    *,
    alpha: float | None = None,
    k: int | None = None,
    seed: int | None = None,
    synthetic_path: str | Path | None = None,
    write: bool = True,
    out_dir: str | Path | None = None,
) -> list[pd.DataFrame]:
    """Partition ``synth_train`` into ``k`` client frames.

    Parameters
    ----------
    strategy:
        ``"iid"`` or ``"label_skew"``.
    alpha:
        Dirichlet concentration. Required when ``strategy="label_skew"``.
    k:
        Number of clients. Defaults to ``cfg.partitioning.n_clients``.
    seed:
        Random seed. Defaults to ``cfg.split.seed``.
    synthetic_path:
        Override for the input parquet. Defaults to ``synth_train.parquet``.
    write:
        When ``True``, write one parquet per client plus metadata.
    out_dir:
        Override for the output directory.

    Returns
    -------
    list[pandas.DataFrame]
        One frame per client, each with ``Customer ID`` + 29 features + target.
    """
    cfg = get_config()
    n_clients = k if k is not None else cfg.partitioning.n_clients
    random_seed = seed if seed is not None else cfg.split.seed

    src_path = (
        Path(synthetic_path) if synthetic_path is not None else cfg.paths.synth_train_parquet
    )
    if not src_path.exists():
        raise FileNotFoundError(
            f"synth_train not found: {src_path}. Run the synthetic split first."
        )

    df = pd.read_parquet(src_path)
    logger.info("Loaded %s (%s rows x %s cols)", src_path, *df.shape)

    if strategy == "iid":
        assignment = assign_iid(df, n_clients, seed=random_seed)
    elif strategy == "label_skew":
        if alpha is None:
            raise ValueError("alpha is required for strategy='label_skew'")
        assignment = assign_dirichlet(df, n_clients, alpha, seed=random_seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    metadata = validate_partitions(df, assignment, n_clients, strategy=strategy)
    if alpha is not None:
        metadata["alpha"] = alpha
    metadata["seed"] = random_seed

    df_with_ids = assign_synthetic_ids(df)
    partitions = [
        df_with_ids.loc[assignment == c].reset_index(drop=True) for c in range(n_clients)
    ]

    # Every partition must carry exactly the expected schema
    expected = ["Customer ID", *FEATURE_COLS, TARGET_COL]
    for c, part in enumerate(partitions):
        assert list(part.columns) == expected, (
            f"Client {c} has unexpected columns: {list(part.columns)}"
        )

    if write:
        target_dir = Path(out_dir) if out_dir is not None else cfg.paths.synthetic_dir
        tag = strategy if alpha is None else f"{strategy}_a{alpha}"
        part_dir = target_dir / f"partitions_{tag}"
        part_dir.mkdir(parents=True, exist_ok=True)

        for c, part in enumerate(partitions):
            path = part_dir / f"client_{c}.parquet"
            part.to_parquet(path, index=False)
        logger.info("Wrote %s client parquet files to %s", n_clients, part_dir)

        meta_path = cfg.paths.partition_metadata
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        existing[tag] = metadata
        meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("Updated %s", meta_path)

    return partitions


def main() -> None:
    """CLI entry point: ``python -m src.partitioning.s05_partition``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    cfg = get_config()

    print("\n" + "=" * 70)
    print("PARTITIONING")
    print("=" * 70)

    for spec in cfg.partitioning.strategies:
        parts = build_partitions(spec.name, alpha=spec.alpha, write=True)
        sizes = [len(p) for p in parts]
        rates = [float(p[TARGET_COL].mean()) for p in parts]
        tag = spec.name if spec.alpha is None else f"{spec.name} a={spec.alpha}"
        print(f"\n  {tag}")
        print(f"    sizes       : {sizes}")
        print(f"    churn rates : {[round(r, 4) for r in rates]}")
        print(f"    total rows  : {sum(sizes):,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
