#!/usr/bin/env python3
"""Environment smoke test.

Verifies that:
  1. All required Python packages import successfully.
  2. The config file loads and paths resolve.
  3. The raw dataset exists and has the expected shape (7043 x 50).
  4. MLflow and MinIO services are reachable (warns, does not fail, if down).

Exit code 0 = all critical checks passed.
"""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# 1. Package imports
# ---------------------------------------------------------------------------
# Tier 1 — required for W1 (clean + EDA). Missing any of these is a hard FAIL.
CORE_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
]

# Tier 2 — required from W2 onwards (amplification, FL, tracking, serving).
# Missing these is a WARN at W1, not a failure.
STACK_PACKAGES = [
    ("copulas", "copulas"),
    ("sdv", "sdv"),
    ("flwr", "flwr"),
    ("torch", "torch"),
    ("ray", "ray"),
    ("opacus", "opacus"),
    ("mlflow", "mlflow"),
    ("fastapi", "fastapi"),
]

OPTIONAL_PACKAGES = [
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("boto3", "boto3"),
]


def _probe(module: str, dist_name: str, tag: str) -> tuple[str, bool]:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "?")
        return f"  [{tag}] {dist_name:<16} {version}", True
    except ImportError:
        return f"  [MISS] {dist_name:<16} NOT INSTALLED", False


def check_imports() -> tuple[list[str], list[str], list[str]]:
    """Return (report lines, missing core, missing stack)."""
    lines: list[str] = []
    missing_core: list[str] = []
    missing_stack: list[str] = []

    lines.append("  -- Tier 1: required for W1 (clean + EDA) --")
    for module, dist_name in CORE_PACKAGES:
        line, ok = _probe(module, dist_name, "OK")
        lines.append(line)
        if not ok:
            missing_core.append(dist_name)

    lines.append("  -- Tier 2: required from W2 onwards --")
    for module, dist_name in STACK_PACKAGES:
        line, ok = _probe(module, dist_name, "OK")
        lines.append(line)
        if not ok:
            missing_stack.append(dist_name)

    lines.append("  -- Optional --")
    for module, dist_name in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            lines.append(f"  [ok] {dist_name:<16} {version} (optional)")
        except ImportError:
            lines.append(f"  [--] {dist_name:<16} not installed (optional)")

    return lines, missing_core, missing_stack


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------
def check_config() -> bool:
    try:
        from src.config import get_config

        cfg = get_config()
        print(f"  [OK]   config loaded: {cfg.project.name}")
        print(f"         raw_data      -> {cfg.paths.raw_data}")
        print(f"         clean_parquet -> {cfg.paths.clean_parquet}")
        print(f"         features      -> {cfg.data.expected_feature_count}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] config: {exc}")
        return False


# ---------------------------------------------------------------------------
# 3. Raw dataset
# ---------------------------------------------------------------------------
def check_dataset() -> bool:
    try:
        import pandas as pd

        from src.config import get_config

        cfg = get_config()
        path = cfg.paths.raw_data
        if not path.exists():
            print(f"  [FAIL] raw dataset missing: {path}")
            return False
        df = pd.read_csv(path, nrows=5)
        n_cols = len(df.columns)
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        status = "OK" if (n_rows == 7043 and n_cols == 50) else "WARN"
        print(f"  [{status}] raw dataset: {n_rows} rows x {n_cols} cols (expected 7043 x 50)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] dataset: {exc}")
        return False


# ---------------------------------------------------------------------------
# 4. Services
# ---------------------------------------------------------------------------
def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_services() -> None:
    for name, host, port in [
        ("MLflow", "localhost", 5000),
        ("MinIO S3 API", "localhost", 9000),
        ("MinIO console", "localhost", 9001),
        ("Postgres", "localhost", 5432),
    ]:
        if _port_open(host, port):
            print(f"  [OK]   {name:<16} {host}:{port}")
        else:
            print(f"  [--]   {name:<16} {host}:{port} not reachable (run: docker compose up -d)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70)

    print("\n[1] Python packages")
    lines, missing_core, missing_stack = check_imports()
    print("\n".join(lines))

    print("\n[2] Configuration")
    cfg_ok = check_config()

    print("\n[3] Raw dataset")
    data_ok = check_dataset()

    print("\n[4] Services")
    check_services()

    print("\n" + "=" * 70)
    if missing_core:
        print(f"RESULT: FAIL — missing W1 packages: {', '.join(missing_core)}")
        print("        fix with: pip install -e \".[dev]\"")
        return 1
    if not (cfg_ok and data_ok):
        print("RESULT: FAIL — config or dataset check failed")
        return 1
    if missing_stack:
        print(f"RESULT: PASS (W1) — stack packages pending for W2+: {', '.join(missing_stack)}")
        print("        install before W2 with: pip install -e \".[dev]\"")
        return 0
    print("RESULT: PASS — full environment ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
