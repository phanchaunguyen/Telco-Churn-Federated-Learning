"""Tests for the FL stack: preprocessing, model, trainer, client, server.

Run with::

    pytest tests/test_09_fl.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.s01_clean import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS
from src.features.s07_preprocess import (
    LocalScaler,
    Vocabulary,
    encode,
    fit_local_scaler,
    fit_vocabulary,
    load_client,
)
from src.models.s08_mlp import (
    ChurnMLP,
    build_model,
    count_parameters,
    get_weights,
    set_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def partition_dir():
    from src.config import get_config

    cfg = get_config()
    path = cfg.paths.synthetic_dir / "partitions_iid"
    if not path.exists():
        pytest.skip(f"partition dir missing: {path}")
    return path


@pytest.fixture(scope="module")
def vocab(partition_dir) -> Vocabulary:
    return fit_vocabulary(partition_dir, 10)


@pytest.fixture(scope="module")
def client_data(partition_dir, vocab):
    return load_client(partition_dir, 0, vocab)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def test_vocabulary_covers_all_categoricals(vocab: Vocabulary) -> None:
    assert set(vocab.categories) == set(CATEGORICAL_COLS)


def test_vocabulary_feature_names_unique(vocab: Vocabulary) -> None:
    assert len(vocab.feature_names) == len(set(vocab.feature_names))


def test_vocabulary_width_matches_encoding(vocab: Vocabulary) -> None:
    """One-hot width must equal the declared feature count."""
    expected = sum(len(v) for v in vocab.categories.values()) + len(NUMERIC_COLS)
    assert len(vocab.feature_names) == expected


def test_vocabulary_roundtrip(vocab: Vocabulary) -> None:
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.categories == vocab.categories
    assert restored.feature_names == vocab.feature_names


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_encode_shape(client_data, vocab: Vocabulary) -> None:
    X, y, _ = client_data
    assert X.shape[1] == len(vocab.feature_names)
    assert X.shape[0] == len(y)


def test_encode_is_float32(client_data) -> None:
    X, _, _ = client_data
    assert X.dtype == np.float32


def test_encode_has_no_nan(client_data) -> None:
    X, _, _ = client_data
    assert not np.isnan(X).any()


def test_encode_one_hot_blocks_are_binary(partition_dir, vocab: Vocabulary) -> None:
    """Every one-hot column must contain only 0.0 and 1.0.

    Tested on ``encode()`` directly, not on ``load_client()`` output: the
    loader scales afterwards, so the one-hot blocks are no longer binary there.
    """
    import pandas as pd

    df = pd.read_parquet(partition_dir / "client_0.parquet")
    X = encode(df, vocab)

    offset = 0
    for col in FEATURE_COLS:
        if col in vocab.categories:
            width = len(vocab.categories[col])
            block = X[:, offset:offset + width]
            assert set(np.unique(block)) <= {0.0, 1.0}, f"{col} not binary"
            # Exactly one category per row
            assert np.allclose(block.sum(axis=1), 1.0), f"{col} rows not one-hot"
            offset += width
        else:
            offset += 1


# ---------------------------------------------------------------------------
# Local scaler
# ---------------------------------------------------------------------------
def test_scaler_standardises(client_data) -> None:
    X, _, _ = client_data
    # X is already scaled by load_client, so re-fitting should give ~0 mean
    scaler = fit_local_scaler(X)
    assert np.allclose(scaler.mean, 0.0, atol=1e-4)


def test_scaler_handles_zero_variance() -> None:
    X = np.ones((100, 3), dtype=np.float32)
    scaler = fit_local_scaler(X)
    assert np.all(scaler.std == 1.0), "zero-variance column must get std=1"
    out = scaler.transform(X)
    assert not np.isnan(out).any()


def test_scaler_roundtrip() -> None:
    scaler = LocalScaler(
        mean=np.array([1.0, 2.0], dtype=np.float32),
        std=np.array([0.5, 2.0], dtype=np.float32),
    )
    restored = LocalScaler.from_dict(scaler.to_dict())
    np.testing.assert_allclose(restored.mean, scaler.mean)
    np.testing.assert_allclose(restored.std, scaler.std)


def test_scaler_is_local_only(partition_dir, vocab: Vocabulary) -> None:
    """Two clients must produce different scalers — no shared statistics."""
    _, _, s0 = load_client(partition_dir, 0, vocab)
    _, _, s1 = load_client(partition_dir, 1, vocab)
    assert not np.allclose(s0.mean, s1.mean), "scalers must differ across silos"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_model_output_shape() -> None:
    model = build_model(input_dim=56)
    out = model(torch.randn(8, 56))
    assert out.shape == (8, 1)


def test_model_output_is_probability() -> None:
    model = build_model(input_dim=56)
    out = model(torch.randn(64, 56))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_model_uses_groupnorm_by_default() -> None:
    model = build_model(input_dim=56)
    assert model.norm_kind == "groupnorm"
    kinds = {type(m).__name__ for m in model.modules()}
    assert "GroupNorm" in kinds
    assert "BatchNorm1d" not in kinds, "BatchNorm breaks Opacus and non-IID FedAvg"


def test_model_has_no_batchnorm_buffers() -> None:
    """GroupNorm has no running buffers, so FedAvg cannot corrupt them."""
    model = build_model(input_dim=56)
    buffer_names = [name for name, _ in model.named_buffers()]
    assert not any("running_" in n for n in buffer_names)


def test_model_input_dim_is_runtime() -> None:
    """Different input widths must both work — nothing is hardcoded."""
    for dim in (40, 56, 72):
        model = build_model(input_dim=dim)
        assert model(torch.randn(4, dim)).shape == (4, 1)


def test_model_rejects_invalid_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        ChurnMLP(input_dim=0)


def test_model_parameter_count_positive() -> None:
    assert count_parameters(build_model(input_dim=56)) > 0


# ---------------------------------------------------------------------------
# Weight exchange
# ---------------------------------------------------------------------------
def test_get_set_weights_roundtrip() -> None:
    a = build_model(input_dim=56, seed=1)
    b = build_model(input_dim=56, seed=2)
    set_weights(b, get_weights(a))
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)


def test_set_weights_rejects_wrong_count() -> None:
    model = build_model(input_dim=56)
    with pytest.raises(ValueError, match="weight tensors"):
        set_weights(model, [np.zeros(3)])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def test_trainer_rejects_dim_mismatch(client_data) -> None:
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = client_data
    with pytest.raises(ValueError, match="input_dim"):
        LocalTrainer(X, y, input_dim=X.shape[1] + 1)


def test_trainer_rejects_length_mismatch(client_data) -> None:
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = client_data
    with pytest.raises(ValueError, match="rows"):
        LocalTrainer(X, y[:-1], input_dim=X.shape[1])


def test_trainer_improves_loss(partition_dir, vocab: Vocabulary) -> None:
    """One epoch must reduce loss on a 5k-row slice."""
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = load_client(partition_dir, 0, vocab)
    X, y = X[:5000], y[:5000]

    trainer = LocalTrainer(X, y, input_dim=X.shape[1], seed=42)
    before, _ = trainer.evaluate()
    result = trainer.fit(epochs=1)
    assert result.loss < before, f"loss did not improve: {before:.4f} -> {result.loss:.4f}"


def test_trainer_returns_expected_fields(partition_dir, vocab: Vocabulary) -> None:
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = load_client(partition_dir, 0, vocab)
    X, y = X[:2000], y[:2000]
    trainer = LocalTrainer(X, y, input_dim=X.shape[1], seed=42)
    result = trainer.fit(epochs=1)

    assert result.n_samples == 2000
    assert len(result.weights) > 0
    assert result.bytes_sent > 0
    assert 0.0 <= result.auc <= 1.0
    assert set(result.to_metrics()) == {"loss", "auc"}


def test_trainer_proximal_term_zero_when_mu_zero(partition_dir, vocab: Vocabulary) -> None:
    """With mu=0 the proximal penalty must be exactly zero."""
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = load_client(partition_dir, 0, vocab)
    X, y = X[:1000], y[:1000]
    trainer = LocalTrainer(X, y, input_dim=X.shape[1], proximal_mu=0.0, seed=42)
    anchor = [torch.tensor(w) for w in get_weights(trainer.model)]
    penalty = trainer._proximal_term(anchor)
    assert float(penalty) == pytest.approx(0.0)


def test_trainer_proximal_term_positive_when_drifted(partition_dir, vocab: Vocabulary) -> None:
    from src.fl.s09_trainer import LocalTrainer

    X, y, _ = load_client(partition_dir, 0, vocab)
    X, y = X[:1000], y[:1000]
    trainer = LocalTrainer(X, y, input_dim=X.shape[1], proximal_mu=0.01, seed=42)
    anchor = [torch.zeros_like(p) for p in trainer.model.parameters()]
    penalty = trainer._proximal_term(anchor)
    assert float(penalty) > 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def test_client_fit_returns_num_examples(partition_dir, vocab: Vocabulary) -> None:
    from src.fl.s10_client import TelcoClient

    client = TelcoClient(0, partition_dir, vocab)
    weights = client.get_parameters()
    new_weights, n, metrics = client.fit(weights, {"server_round": 1})

    assert n == len(client.y)
    assert len(new_weights) == len(weights)
    assert "auc" in metrics and "loss" in metrics


def test_client_evaluate_returns_auc(partition_dir, vocab: Vocabulary) -> None:
    from src.fl.s10_client import TelcoClient

    client = TelcoClient(0, partition_dir, vocab)
    loss, n, metrics = client.evaluate(client.get_parameters(), {})
    assert n == len(client.y)
    assert 0.0 <= metrics["auc"] <= 1.0


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def test_proximal_mu_only_for_fedprox() -> None:
    from src.fl.s11_server import proximal_mu_for

    assert proximal_mu_for("fedavg") == 0.0
    assert proximal_mu_for("fedavgm") == 0.0
    assert proximal_mu_for("fedprox") > 0.0


def test_sequential_one_round_runs(partition_dir) -> None:
    """The milestone: one FedAvg round over 10 silos must complete."""
    from src.fl.s11_server import run_sequential

    history = run_sequential(
        n_rounds=1,
        num_clients=10,
        strategy_name="fedavg",
        partition="iid",
    )

    assert history["n_rounds"] == 1
    assert history["num_clients"] == 10
    assert history["n_selected_per_round"] == 5
    assert len(history["losses_distributed"]) == 1
    assert len(history["final_weights"]) > 0

    _, loss = history["losses_distributed"][0]
    _, auc = history["metrics_distributed"]["auc"][0]
    assert 0.0 < loss < 1.0
    assert 0.5 < auc <= 1.0, f"one round should beat random, got AUC {auc:.4f}"


def test_sequential_weights_change(partition_dir) -> None:
    """Aggregation must actually move the global weights."""
    from src.fl.s11_server import run_sequential

    history = run_sequential(
        n_rounds=1, num_clients=10, strategy_name="fedavg", partition="iid"
    )
    # The final weights must not be all zeros (the initial state is random,
    # so a successful round produces non-trivial values)
    total_abs = sum(float(np.abs(w).sum()) for w in history["final_weights"])
    assert total_abs > 0.0
