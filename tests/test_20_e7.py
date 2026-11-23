"""Tests for :mod:`src.experiments.s20_e7_communication`.

The full sweep takes ~18 minutes, so only the pure helpers are tested here.

Run with::

    pytest tests/test_20_e7.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.s20_e7_communication import (
    BYTES_PER_PARAM,
    DEFAULT_FRACTIONS,
    DIRECTIONS,
    _render_e7,
    bytes_per_round,
)


# ---------------------------------------------------------------------------
# bytes_per_round
# ---------------------------------------------------------------------------
def test_bytes_per_round_matches_hand_calculation() -> None:
    """2 directions x 18,049 params x 4 bytes x 5 clients = 721,960 bytes."""
    assert bytes_per_round(18049, 5) == 721_960


def test_bytes_per_round_scales_with_clients() -> None:
    one = bytes_per_round(18049, 1)
    ten = bytes_per_round(18049, 10)

    assert ten == one * 10


def test_bytes_per_round_zero_clients() -> None:
    assert bytes_per_round(18049, 0) == 0


def test_constants_match_the_documented_formula() -> None:
    assert BYTES_PER_PARAM == 4
    assert DIRECTIONS == 2


def test_default_fractions_match_protocol() -> None:
    assert DEFAULT_FRACTIONS == [0.25, 0.5, 1.0]


def test_protocol_expected_values() -> None:
    """The three protocol rows: 0.25 -> 2 clients, 0.5 -> 5, 1.0 -> 10."""
    n_params = 18049

    assert bytes_per_round(n_params, 2) == 288_784
    assert bytes_per_round(n_params, 5) == 721_960
    assert bytes_per_round(n_params, 10) == 1_443_920


# ---------------------------------------------------------------------------
# _render_e7
# ---------------------------------------------------------------------------
def _table():
    return pd.DataFrame([
        {
            "fraction": 0.25,
            "n_selected_per_round": 2,
            "mb_per_round": 0.2888,
            "cumulative_mb": 14.44,
            "auc_real": 0.8853,
            "pr_auc_real": 0.7800,
            "recall_at_10pct_real": 0.3100,
            "auc_synth": 0.9200,
        },
        {
            "fraction": 1.0,
            "n_selected_per_round": 10,
            "mb_per_round": 1.4439,
            "cumulative_mb": 72.20,
            "auc_real": 0.8872,
            "pr_auc_real": 0.7850,
            "recall_at_10pct_real": 0.3150,
            "auc_synth": 0.9210,
        },
    ])


def _runs():
    return [{"n_params": 18049}]


def test_render_e7_contains_every_fraction() -> None:
    md = _render_e7(_table(), _runs(), 50)

    assert "0.25" in md
    assert "1.00" in md


def test_render_e7_shows_the_formula() -> None:
    md = _render_e7(_table(), _runs(), 50)

    assert "18,049" in md
    assert "bytes_per_round" in md


def test_render_e7_reports_pass_when_cheap_setting_is_close() -> None:
    md = _render_e7(_table(), _runs(), 50)

    assert "PASS" in md


def test_render_e7_reports_fail_when_cheap_setting_is_far() -> None:
    table = _table()
    table.loc[table["fraction"] == 0.25, "auc_real"] = 0.80
    md = _render_e7(table, _runs(), 50)

    assert "FAIL" in md


def test_render_e7_explains_the_factor_of_two() -> None:
    md = _render_e7(_table(), _runs(), 50)

    assert "factor of 2" in md
    assert "download" in md


def test_render_e7_documents_the_fraction_floor() -> None:
    """fraction=0.25 gives 2 clients, not the 2-3 the protocol sketched."""
    md = _render_e7(_table(), _runs(), 50)

    assert "max(2" in md


def test_render_e7_single_row_does_not_crash() -> None:
    md = _render_e7(_table().head(1), _runs(), 50)

    assert "E7" in md


def test_render_e7_empty_runs_does_not_crash() -> None:
    md = _render_e7(_table(), [], 50)

    assert "E7" in md
