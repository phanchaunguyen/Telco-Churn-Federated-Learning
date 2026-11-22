"""Local MLP for federated churn prediction.

Architecture (input dimension is RUNTIME-DETERMINED, never hardcoded)::

    Input (D) -> Linear(D, 128) -> GroupNorm -> ReLU -> Dropout(0.3)
              -> Linear(128, 64) -> GroupNorm -> ReLU -> Dropout(0.3)
              -> Linear(64, 32)  -> ReLU
              -> Linear(32, 1)   -> Sigmoid

Why GroupNorm instead of BatchNorm
----------------------------------
Three independent reasons, any one of which is sufficient:

1. **Opacus cannot wrap BatchNorm.** DP-SGD needs per-sample gradients, but a
   BatchNorm output depends on every other sample in the batch through the
   batch mean, so the per-sample gradient is not well defined. Opacus raises
   ``UnsupportedModuleError``. E4 (privacy-utility) is impossible without this
   change.

2. **Fair comparison across E1-E7.** If E1-E3 used BatchNorm and E4 used
   GroupNorm, any AUC difference between them would confound the privacy
   setting with the architecture change.

3. **BatchNorm breaks under non-IID FedAvg.** ``state_dict()`` includes the
   ``running_mean`` / ``running_var`` buffers, so FedAvg averages them across
   clients. At alpha=0.1 the silos have wildly different feature distributions
   (client 5 is 0.87% churn, client 8 is 91.2%), so the averaged buffer matches
   no client. GroupNorm has no buffers, so the problem does not arise.

Usage
-----
>>> from src.models.s08_mlp import build_model
>>> model = build_model(input_dim=56)
>>> model(torch.randn(8, 56)).shape
torch.Size([8, 1])
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Building blocks
# =============================================================================
def _norm_layer(kind: str, num_features: int, num_groups: int) -> nn.Module:
    """Return the requested normalisation layer.

    ``groupnorm`` is the default and the only option compatible with Opacus.
    ``batchnorm`` exists so the choice is explicit rather than implicit.
    """
    if kind == "groupnorm":
        # GroupNorm requires num_features % num_groups == 0
        groups = min(num_groups, num_features)
        while num_features % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_features)
    if kind == "batchnorm":
        return nn.BatchNorm1d(num_features)
    if kind == "layernorm":
        return nn.LayerNorm(num_features)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind!r}")


# =============================================================================
# Model
# =============================================================================
class ChurnMLP(nn.Module):
    """Binary churn classifier.

    Parameters
    ----------
    input_dim:
        Number of features after one-hot encoding. Read from
        ``X_train.shape[1]`` at runtime — never hardcoded.
    hidden_dims:
        Widths of the hidden layers.
    dropout:
        Dropout probability after each normalised hidden layer.
    norm:
        ``groupnorm`` (default), ``batchnorm``, ``layernorm``, or ``none``.
    num_groups:
        Groups for GroupNorm. Clamped so it divides the layer width.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        norm: str = "groupnorm",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")

        dims = hidden_dims or [128, 64, 32]
        self.input_dim = input_dim
        self.hidden_dims = list(dims)
        self.norm_kind = norm

        layers: list[nn.Module] = []
        prev = input_dim

        # Hidden layers 0 and 1 get norm + activation + dropout
        for i, width in enumerate(dims):
            layers.append(nn.Linear(prev, width))
            if i < len(dims) - 1:
                layers.append(_norm_layer(norm, width, num_groups))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                # Final hidden layer: activation only, no norm, no dropout
                layers.append(nn.ReLU())
            prev = width

        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        logger.info(
            "ChurnMLP: input_dim=%s hidden=%s norm=%s dropout=%s params=%s",
            input_dim,
            dims,
            norm,
            dropout,
            f"{sum(p.numel() for p in self.parameters()):,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return churn probability of shape ``(batch, 1)``."""
        return self.net(x)


# =============================================================================
# Factory
# =============================================================================
def build_model(
    input_dim: int,
    *,
    hidden_dims: list[int] | None = None,
    dropout: float | None = None,
    norm: str | None = None,
    num_groups: int | None = None,
    seed: int | None = None,
) -> ChurnMLP:
    """Build a model using ``configs/base.yaml`` defaults where not overridden."""
    cfg = get_config()
    model_cfg = cfg.model

    if seed is not None:
        torch.manual_seed(seed)

    return ChurnMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims if hidden_dims is not None else model_cfg.hidden_dims,
        dropout=dropout if dropout is not None else model_cfg.dropout,
        norm=norm if norm is not None else model_cfg.norm,
        num_groups=num_groups if num_groups is not None else model_cfg.num_groups,
    )


# =============================================================================
# Weight helpers (Flower exchanges numpy arrays)
# =============================================================================
def get_weights(model: nn.Module) -> list[np.ndarray]:  # noqa: F821
    """Return model parameters as a list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:  # noqa: F821
    """Load numpy arrays into the model's state dict."""
    state = model.state_dict()
    if len(weights) != len(state):
        raise ValueError(
            f"Expected {len(state)} weight tensors, got {len(weights)}"
        )
    new_state = {
        key: torch.tensor(value, dtype=state[key].dtype)
        for key, value in zip(state.keys(), weights, strict=True)
    }
    model.load_state_dict(new_state)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    """CLI entry point: ``python -m src.models.s08_mlp``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    model = build_model(input_dim=56)
    x = torch.randn(8, 56)
    y = model(x)

    print("\n" + "=" * 70)
    print("MLP SMOKE TEST")
    print("=" * 70)
    print(f"  input_dim   : 56")
    print(f"  output shape: {tuple(y.shape)}")
    print(f"  output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"  parameters  : {count_parameters(model):,}")
    print(f"  norm        : {model.norm_kind}")
    print("=" * 70)


if __name__ == "__main__":
    main()
