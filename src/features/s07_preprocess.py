"""Per-client preprocessing: one-hot encoding and local-only scaling.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION → PREPROCESS**

Privacy constraint
------------------
Every client fits its own scaler on its own silo. A global mean/std would leak
information across silos — the whole point of FL is that no client sees another
client's data, and that includes aggregate statistics.

The consequence is that the same feature has a different scale in each silo.
That is expected and correct: the model learns weights that work on the
normalised representation, and FedAvg averages those weights, not the raw data.

Input dimension
---------------
The one-hot width depends on which categories appear in a silo. A silo that
never saw ``Offer = "None"`` would produce a narrower matrix than one that did,
and FedAvg cannot average weight tensors of different shapes.

The fix is to fit the category vocabulary on the **union** of categories seen
across all silos, then apply that fixed vocabulary to every silo. The vocabulary
is metadata (which category names exist), not data (how many customers have
each), so sharing it does not leak anything.

Usage
-----
>>> from src.features.s07_preprocess import load_client, fit_vocabulary
>>> vocab = fit_vocabulary(partition_dir, k=10)
>>> X, y, scaler = load_client(partition_dir, 0, vocab)
>>> X.shape
(90000, 56)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.s01_clean import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, TARGET_COL

logger = logging.getLogger(__name__)


# =============================================================================
# Vocabulary — shared category names, no data
# =============================================================================
@dataclass
class Vocabulary:
    """Fixed one-hot column layout shared by every client.

    Attributes
    ----------
    categories:
        ``{column: [category, ...]}`` in a deterministic order.
    feature_names:
        The resulting column names after one-hot expansion, in order.
    """

    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"categories": self.categories, "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Vocabulary":
        return cls(
            categories=payload["categories"],
            feature_names=payload["feature_names"],
        )


def fit_vocabulary(partition_dir: str | Path, k: int) -> Vocabulary:
    """Collect the union of category values across all client silos.

    Only the set of category *names* is collected — never counts, never
    distributions. This is metadata, so it is safe to share.
    """
    part_dir = Path(partition_dir)
    seen: dict[str, set[str]] = {col: set() for col in CATEGORICAL_COLS}

    for cid in range(k):
        path = part_dir / f"client_{cid}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing client silo: {path}")
        df = pd.read_parquet(path, columns=CATEGORICAL_COLS)
        for col in CATEGORICAL_COLS:
            seen[col].update(df[col].astype(str).unique())

    categories = {col: sorted(values) for col, values in seen.items()}

    feature_names: list[str] = []
    for col in FEATURE_COLS:
        if col in categories:
            feature_names.extend(f"{col}={v}" for v in categories[col])
        else:
            feature_names.append(col)

    vocab = Vocabulary(categories=categories, feature_names=feature_names)
    logger.info(
        "Vocabulary: %s categorical columns -> %s one-hot features",
        len(categories),
        len(feature_names),
    )
    return vocab


# =============================================================================
# Per-client scaler
# =============================================================================
@dataclass
class LocalScaler:
    """Standard scaler fitted on ONE client's silo.

    Deliberately not ``sklearn.StandardScaler``: keeping it explicit makes it
    obvious in code review that the statistics come from a single silo.
    """

    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def fit_local_scaler(X: np.ndarray) -> LocalScaler:
    """Fit mean/std on a single client's feature matrix.

    Zero-variance columns get ``std = 1`` so the transform is a no-op rather
    than a division by zero.
    """
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return LocalScaler(mean=mean, std=std)


# =============================================================================
# Encoding
# =============================================================================
def encode(df: pd.DataFrame, vocab: Vocabulary) -> np.ndarray:
    """One-hot encode categoricals and stack numerics, using a fixed layout."""
    n = len(df)
    blocks: list[np.ndarray] = []

    for col in FEATURE_COLS:
        if col in vocab.categories:
            values = df[col].astype(str).to_numpy()
            for category in vocab.categories[col]:
                blocks.append((values == category).astype(np.float32).reshape(n, 1))
        else:
            blocks.append(df[col].to_numpy(dtype=np.float32).reshape(n, 1))

    return np.hstack(blocks)


# =============================================================================
# Loading
# =============================================================================
def load_client(
    partition_dir: str | Path,
    client_id: int,
    vocab: Vocabulary,
    *,
    fit_scaler: bool = True,
    scaler: LocalScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, LocalScaler]:
    """Load one silo, encode it, and scale it with local statistics.

    Parameters
    ----------
    partition_dir:
        Directory holding ``client_{i}.parquet``.
    client_id:
        Which silo to load.
    vocab:
        Shared one-hot layout.
    fit_scaler:
        When ``True`` (default) fit a scaler on this silo. Set to ``False`` and
        pass ``scaler`` to reuse a previously fitted one (evaluation, serving).
    scaler:
        Pre-fitted scaler to apply when ``fit_scaler=False``.

    Returns
    -------
    (X, y, scaler)
        ``X`` is float32 ``(n, d)``, ``y`` is int64 ``(n,)``.
    """
    path = Path(partition_dir) / f"client_{client_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing client silo: {path}")

    df = pd.read_parquet(path)
    X = encode(df, vocab)
    y = df[TARGET_COL].to_numpy(dtype=np.int64)

    if fit_scaler:
        local_scaler = fit_local_scaler(X)
    else:
        if scaler is None:
            raise ValueError("scaler is required when fit_scaler=False")
        local_scaler = scaler

    X = local_scaler.transform(X).astype(np.float32)

    assert X.shape[1] == len(vocab.feature_names), (
        f"encoded width {X.shape[1]} != vocabulary width {len(vocab.feature_names)}"
    )
    assert not np.isnan(X).any(), f"client {client_id} produced NaN features"

    logger.info(
        "Loaded client %s: X %s, y %s, churn %.4f",
        client_id,
        X.shape,
        y.shape,
        float(y.mean()),
    )
    return X, y, local_scaler


def load_all_clients(
    partition_dir: str | Path,
    k: int,
    vocab: Vocabulary,
) -> list[tuple[np.ndarray, np.ndarray, LocalScaler]]:
    """Load every silo. Each gets its own scaler."""
    return [load_client(partition_dir, cid, vocab) for cid in range(k)]


# =============================================================================
# Persistence
# =============================================================================
def save_preprocessing(
    vocab: Vocabulary,
    scalers: dict[int, LocalScaler],
    out_path: str | Path,
) -> None:
    """Persist the vocabulary and per-client scalers for serving reuse."""
    payload = {
        "vocabulary": vocab.to_dict(),
        "scalers": {str(cid): s.to_dict() for cid, s in scalers.items()},
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


def load_preprocessing(path: str | Path) -> tuple[Vocabulary, dict[int, LocalScaler]]:
    """Load a previously saved vocabulary and scaler set."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = Vocabulary.from_dict(payload["vocabulary"])
    scalers = {
        int(cid): LocalScaler.from_dict(s) for cid, s in payload["scalers"].items()
    }
    return vocab, scalers


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.features.s07_preprocess``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    cfg = get_config()
    part_dir = cfg.paths.synthetic_dir / "partitions_iid"
    k = cfg.partitioning.n_clients

    vocab = fit_vocabulary(part_dir, k)
    clients = load_all_clients(part_dir, k, vocab)
    scalers = {cid: s for cid, (_, _, s) in enumerate(clients)}

    out = cfg.paths.processed_dir / "preprocessing_iid.json"
    save_preprocessing(vocab, scalers, out)

    print("\n" + "=" * 70)
    print("PREPROCESS COMPLETE")
    print("=" * 70)
    print(f"  partition dir   : {part_dir}")
    print(f"  clients         : {k}")
    print(f"  input dim       : {len(vocab.feature_names)}")
    print(f"  categorical cols: {len(vocab.categories)}")
    print(f"  numeric cols    : {len(NUMERIC_COLS)}")
    print(f"  X shape         : {clients[0][0].shape}")
    print(f"  saved           : {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
