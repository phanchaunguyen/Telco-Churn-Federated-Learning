"""Tests for :mod:`src.experiments.s19_e6_scalability`.

The full sweep takes ~50 minutes, so only the pure helpers are tested here.

Run with::

    pytest tests/test_19_e6.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.s19_e6_scalability import (
    DEFAULT_KS,
    _render_e6,
    partition_dir_for,
)


# ---------------------------------------------------------------------------
# partition_dir_for
# ---------------------------------------------------------------------------
def test_partition_dir_for_k10_uses_legacy_name() -> None:
    """K=10 must keep ``partitions_iid`` so E1-E5 stay reproducible."""
    path = partition_dir_for(10)

    assert path.name == "partitions_iid"


def test_partition_dir_for_other_k_appends_suffix() -> None:
    for k in (5, 20, 50):
        path = partition_dir_for(k)
        assert path.name == f"partitions_iid_k{k}"


def test_partition_dir_for_missing_k_raises() -> None:
    with pytest.raises(FileNotFoundError, match="build_partitions"):
        partition_dir_for(999)


def test_default_ks_matches_protocol() -> None:
    assert DEFAULT_KS == [5, 10, 20, 50]


# ---------------------------------------------------------------------------
# _render_e6
# ---------------------------------------------------------------------------
def _table():
    return pd.DataFrame([
        {
            "k": 5,
            "n_selected_per_round": 2,
            "total_seconds": 354.4,
            "sec_per_round": 7.09,
            "sec_per_client_round": 3.544,
            "sanity_auc_synth": 0.8012,
        },
        {
            "k": 10,
            "n_selected_per_round": 5,
            "total_seconds": 381.1,
            "sec_per_round": 7.62,
            "sec_per_client_round": 1.524,
            "sanity_auc_synth": 0.8007,
        },
        {
            "k": 50,
            "n_selected_per_round": 25,
            "total_seconds": 387.1,
            "sec_per_round": 7.74,
            "sec_per_client_round": 0.310,
            "sanity_auc_synth": 0.7960,
        },
    ])


def test_render_e6_contains_every_k() -> None:
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0, "k10_growth_ratio": 1.0}, 50)

    for k in ("5", "10", "50"):
        assert f"| {k} |" in md


def test_render_e6_reports_pass_when_flat() -> None:
    """Flat wall-clock across K is the expected outcome, not a failure."""
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0}, 50)
    assert "PASS" in md


def test_render_e6_reports_fail_when_time_explodes() -> None:
    """A 3x jump in total time across K is not flat and must fail."""
    table = _table()
    table.loc[table["k"] == 50, "total_seconds"] = 2000.0
    md = _render_e6(table, {"k5_growth_ratio": 1.0}, 50)
    assert "FAIL" in md


def test_render_e6_explains_why_cost_is_flat() -> None:
    """The report must explain the mechanism, not just report a number."""
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0}, 50)

    assert "fraction_fit" in md
    assert "Rows per round" in md


def test_render_e6_shows_rows_per_round_table() -> None:
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0}, 50)

    # K=5 -> 2 clients x 180,000 rows = 360,000
    assert "360,000" in md


def test_render_e6_states_auc_is_only_a_sanity_check() -> None:
    """The report must not let the AUC column read as a quality claim."""
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0}, 50)

    assert "sanity check" in md
    assert "not a" in md


def test_render_e6_explains_iid_choice() -> None:
    md = _render_e6(_table(), {"k5_growth_ratio": 1.0}, 50)

    assert "IID" in md
    assert "confound" in md


def test_render_e6_single_k_does_not_crash() -> None:
    md = _render_e6(_table().head(1), {"k5_growth_ratio": 1.0}, 50)

    assert "E6" in md
