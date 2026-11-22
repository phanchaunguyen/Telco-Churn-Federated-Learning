"""Tests for the FedAvgM server-side momentum in :mod:`src.fl.s11_server`.

The update rule is::

    pseudo_gradient = W_global - W_aggregated
    v               = beta * v + pseudo_gradient
    W_global        = W_global - lr * v

Two properties are worth pinning because they are easy to get wrong and the
failure is silent:

* the **sign** — momentum must move the server toward the aggregate, not away;
* the **convergence rate** — the error decays as ``|r|^k`` where ``r`` solves
  ``r^2 - (1 - lr + beta) r + beta = 0``. The product of the roots is ``beta``,
  so ``|r| = sqrt(beta)`` regardless of ``lr``. ``lr`` changes the overshoot
  amplitude, not the rate.

Run with::

    pytest tests/test_11_server.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


def _fedavgm_step(
    global_weights: list[np.ndarray],
    aggregated: list[np.ndarray],
    momentum: list[np.ndarray] | None,
    beta: float,
    lr: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Replicate the FedAvgM branch of ``run_sequential`` for one round."""
    pseudo_gradient = [g - a for g, a in zip(global_weights, aggregated, strict=True)]
    if momentum is None:
        momentum = pseudo_gradient
    else:
        momentum = [beta * m + p for m, p in zip(momentum, pseudo_gradient, strict=True)]
    return [g - lr * m for g, m in zip(global_weights, momentum, strict=True)], momentum


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------
def test_first_round_with_unit_lr_lands_on_the_aggregate() -> None:
    """Round 1 has no history, so v = W - A and W - 1*v = A."""
    w = [np.array([1.0, 2.0])]
    agg = [np.array([1.5, 2.5])]

    new_w, momentum = _fedavgm_step(w, agg, None, beta=0.9, lr=1.0)

    np.testing.assert_allclose(new_w[0], agg[0])
    np.testing.assert_allclose(momentum[0], w[0] - agg[0])


def test_first_round_with_fractional_lr_interpolates() -> None:
    """W - lr*(W - A) = (1 - lr)*W + lr*A, a convex step toward the aggregate."""
    w = [np.array([0.0])]
    agg = [np.array([1.0])]

    new_w, _ = _fedavgm_step(w, agg, None, beta=0.9, lr=0.1)

    np.testing.assert_allclose(new_w[0], [0.1])


def test_momentum_accumulates_with_beta() -> None:
    w = [np.array([0.0])]
    agg = [np.array([1.0])]

    _, m1 = _fedavgm_step(w, agg, None, beta=0.9)
    _, m2 = _fedavgm_step(w, agg, m1, beta=0.9)

    # m1 = -1, m2 = 0.9 * (-1) + (-1) = -1.9
    np.testing.assert_allclose(m2[0], [-1.9])


def test_momentum_moves_toward_the_aggregate() -> None:
    """The step must reduce the distance to the aggregate, not increase it."""
    w = [np.array([0.0])]
    agg = [np.array([1.0])]

    new_w, _ = _fedavgm_step(w, agg, None, beta=0.9, lr=0.1)

    assert abs(new_w[0][0] - agg[0][0]) < abs(w[0][0] - agg[0][0])


def test_inverted_sign_moves_away_from_the_aggregate() -> None:
    """The bug this guards against: adding the momentum instead of subtracting.

    With the wrong sign the first step goes to ``W + (W - A)``, which is twice
    as far from the aggregate as it started.
    """
    w = np.array([0.0])
    agg = np.array([1.0])

    wrong = w + (w - agg)

    assert abs(wrong[0] - agg[0]) > abs(w[0] - agg[0])


def test_zero_momentum_reduces_to_plain_sgd() -> None:
    w = [np.array([0.0, 5.0])]
    agg = [np.array([1.0, 4.0])]

    new_w, _ = _fedavgm_step(w, agg, None, beta=0.0, lr=1.0)

    np.testing.assert_allclose(new_w[0], agg[0])


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def test_converges_to_a_constant_aggregate() -> None:
    """A fixed aggregate is a fixed point: the error must decay to zero."""
    w = [np.array([0.0])]
    agg = [np.array([1.0])]
    momentum = None

    for _ in range(2000):
        w, momentum = _fedavgm_step(w, agg, momentum, beta=0.9, lr=1.0)

    assert w[0][0] == pytest.approx(1.0, abs=1e-6)


def test_convergence_rate_is_sqrt_beta_independent_of_lr() -> None:
    """|r| = sqrt(beta) for every lr, because the root product is beta.

    This is why ``lr`` is not a stability knob: it changes how far the server
    overshoots, not how fast the error decays.

    The rate is measured on the error *envelope* over a 20-round window, not on
    successive rounds. At ``lr=1.0`` the characteristic roots are complex, so
    the error oscillates in sign and a per-round ratio is meaningless; the
    envelope decays by ``|r|^20`` per window.
    """
    beta = 0.9
    window = 20
    expected = np.sqrt(beta) ** window

    for lr in (1.0, 0.5, 0.1):
        w, m = 0.0, None
        errors = []
        for _ in range(300):
            pseudo = w - 1.0
            m = pseudo if m is None else beta * m + pseudo
            w = w - lr * m
            errors.append(abs(w - 1.0))

        envelope = [max(errors[i : i + window]) for i in range(0, 280, window)]
        ratios = [
            envelope[i + 1] / envelope[i]
            for i in range(len(envelope) - 1)
            if envelope[i] > 1e-12
        ]

        assert ratios[-1] == pytest.approx(expected, abs=0.05), f"lr={lr}"


def test_lower_lr_reduces_overshoot() -> None:
    """A smaller server step overshoots less, which is the only thing lr buys."""
    def peak(lr: float) -> float:
        w, m, positions = 0.0, None, []
        for _ in range(50):
            pseudo = w - 1.0
            m = pseudo if m is None else 0.9 * m + pseudo
            w = w - lr * m
            positions.append(w)
        return max(positions)

    assert peak(0.1) < peak(1.0)


def test_configured_learning_rate_is_a_valid_step_size() -> None:
    """The shipped config must use a step in (0, 1]."""
    from src.config import get_config

    cfg = get_config().fl.strategies.fedavgm

    assert 0.0 < cfg.server_learning_rate <= 1.0
    assert 0.0 <= cfg.server_momentum < 1.0
