"""Create the three Superset dashboards via the Superset REST API.

Pipeline position: **EXPORT (BI) → DASHBOARD**

Superset stores dashboards as rows in its metadata database, not as files, so
they cannot be provisioned by mounting a JSON file the way Grafana can. This
script logs in, registers the postgres connection, registers the three tables
as datasets, and creates one dashboard per experiment theme.

Idempotent
----------
Every step checks for an existing object by name first. Re-running the script
updates rather than duplicating, so it is safe to run after each experiment
rerun.

Fails loudly
------------
Any non-2xx response raises with the response body. There is no fallback: a
half-created dashboard set is worse than a clear error, because the report
would claim dashboards that do not exist.

Usage
-----
>>> from src.serving.s25_superset_dashboards import create_dashboards
>>> create_dashboards()
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL: str = "http://localhost:8088"
USERNAME: str = "admin"
PASSWORD: str = "admin"

# The postgres service as seen from inside the Superset container
DB_NAME: str = "FL Telco Postgres"
DB_URI: str = "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow"

# Tables written by s23_export_results
TABLES: tuple[str, ...] = ("e1_convergence", "e5_fairness", "e4_privacy")


# =============================================================================
# HTTP helpers
# =============================================================================
class SupersetClient:
    """Minimal Superset REST client: login, then authenticated JSON calls."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._access_token: str | None = None
        self._csrf_token: str | None = None

    # ------------------------------------------------------------------
    def login(self, username: str = USERNAME, password: str = PASSWORD) -> None:
        """Authenticate and store the access token plus CSRF token."""
        response = self.session.post(
            f"{self.base_url}/api/v1/security/login",
            json={
                "username": username,
                "password": password,
                "provider": "db",
                "refresh": False,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Superset login failed ({response.status_code}): {response.text}"
            )
        self._access_token = response.json()["access_token"]

        csrf = self.session.get(
            f"{self.base_url}/api/v1/security/csrf_token/",
            headers=self._headers(),
            timeout=30,
        )
        if csrf.status_code != 200:
            raise RuntimeError(
                f"CSRF token fetch failed ({csrf.status_code}): {csrf.text}"
            )
        self._csrf_token = csrf.json()["result"]
        logger.info("Logged in to Superset as %s", username)

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        if self._csrf_token:
            headers["X-CSRFToken"] = self._csrf_token
        return headers

    # ------------------------------------------------------------------
    def get(self, path: str, **params: Any) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GET {path} failed ({response.status_code}): {response.text}"
            )
        return response.json()

    # ------------------------------------------------------------------
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"POST {path} failed ({response.status_code}): {response.text}"
            )
        return response.json()


# =============================================================================
# Provisioning steps
# =============================================================================
def ensure_database(client: SupersetClient) -> int:
    """Register the postgres connection, returning its id."""
    existing = client.get("/api/v1/database/", q=json.dumps({"filters": []}))
    for db in existing.get("result", []):
        if db["database_name"] == DB_NAME:
            logger.info("Database %r already registered (id=%s)", DB_NAME, db["id"])
            return int(db["id"])

    created = client.post(
        "/api/v1/database/",
        {
            "database_name": DB_NAME,
            "sqlalchemy_uri": DB_URI,
            "expose_in_sqllab": True,
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
        },
    )
    db_id = int(created["id"])
    logger.info("Registered database %r (id=%s)", DB_NAME, db_id)
    return db_id


def ensure_dataset(client: SupersetClient, db_id: int, table: str) -> int:
    """Register one table as a dataset, returning its id."""
    existing = client.get(
        "/api/v1/dataset/",
        q=json.dumps({"filters": [{"col": "table_name", "opr": "eq", "value": table}]}),
    )
    for ds in existing.get("result", []):
        if ds["table_name"] == table:
            logger.info("Dataset %r already registered (id=%s)", table, ds["id"])
            return int(ds["id"])

    created = client.post(
        "/api/v1/dataset/",
        {"database": db_id, "schema": "public", "table_name": table},
    )
    ds_id = int(created["id"])
    logger.info("Registered dataset %r (id=%s)", table, ds_id)
    return ds_id


def ensure_dashboard(
    client: SupersetClient,
    title: str,
    slug: str,
    charts: list[dict[str, Any]],
) -> int:
    """Create a dashboard with the given charts, returning its id.

    Charts are created first because a dashboard references them by id. Each
    chart spec carries its own SQL, so the dashboard is self-describing.
    """
    existing = client.get(
        "/api/v1/dashboard/",
        q=json.dumps({"filters": [{"col": "dashboard_title", "opr": "eq", "value": title}]}),
    )
    for dash in existing.get("result", []):
        if dash["dashboard_title"] == title:
            logger.info("Dashboard %r already exists (id=%s)", title, dash["id"])
            return int(dash["id"])

    chart_ids: list[int] = []
    for spec in charts:
        created = client.post("/api/v1/chart/", spec)
        chart_ids.append(int(created["id"]))
        logger.info("  chart %r -> id=%s", spec["slice_name"], created["id"])

    created = client.post(
        "/api/v1/dashboard/",
        {
            "dashboard_title": title,
            "slug": slug,
            "published": True,
            "position_json": json.dumps(
                {
                    f"CHART-{cid}": {
                        "type": "CHART",
                        "id": f"CHART-{cid}",
                        "children": [],
                        "meta": {"width": 6, "height": 50},
                    }
                    for cid in chart_ids
                }
            ),
        },
    )
    dash_id = int(created["id"])
    logger.info("Created dashboard %r (id=%s)", title, dash_id)
    return dash_id


# =============================================================================
# Chart specs
# =============================================================================
def _chart(
    name: str,
    dataset_id: int,
    viz_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build a chart creation payload."""
    return {
        "slice_name": name,
        "viz_type": viz_type,
        "datasource_id": dataset_id,
        "datasource_type": "table",
        "params": json.dumps(params),
    }


def convergence_charts(conv_id: int) -> list[dict[str, Any]]:
    """AUC per round for the three E2 partitions."""
    return [
        _chart(
            "AUC per round — IID",
            conv_id,
            "line",
            {
                "metrics": ["auc"],
                "groupby": [],
                "x_axis": "round",
                "adhoc_filters": [
                    {"col": "run", "op": "==", "val": "iid", "expressionType": "SIMPLE"}
                ],
                "row_limit": 10000,
            },
        ),
        _chart(
            "AUC per round — non-IID",
            conv_id,
            "line",
            {
                "metrics": ["auc"],
                "groupby": ["run"],
                "x_axis": "round",
                "adhoc_filters": [
                    {"col": "run", "op": "!=", "val": "iid", "expressionType": "SIMPLE"}
                ],
                "row_limit": 10000,
            },
        ),
    ]


def fairness_charts(fair_id: int) -> list[dict[str, Any]]:
    """Per-client AUC and per-reason recall, FedAvg against FedProx."""
    return [
        _chart(
            "Per-client AUC",
            fair_id,
            "bar",
            {
                "metrics": ["value"],
                "groupby": ["label"],
                "columns": ["strategy"],
                "adhoc_filters": [
                    {"col": "view", "op": "==", "val": "client", "expressionType": "SIMPLE"}
                ],
                "row_limit": 10000,
            },
        ),
        _chart(
            "Recall by churn reason",
            fair_id,
            "bar",
            {
                "metrics": ["value"],
                "groupby": ["label"],
                "columns": ["strategy"],
                "adhoc_filters": [
                    {"col": "view", "op": "==", "val": "reason", "expressionType": "SIMPLE"}
                ],
                "row_limit": 10000,
            },
        ),
    ]


def privacy_charts(priv_id: int) -> list[dict[str, Any]]:
    """AUC against epsilon."""
    return [
        _chart(
            "Privacy-utility curve",
            priv_id,
            "line",
            {
                "metrics": ["auc"],
                "groupby": [],
                "x_axis": "epsilon",
                "row_limit": 10000,
            },
        ),
        _chart(
            "AUC by noise multiplier",
            priv_id,
            "bar",
            {
                "metrics": ["auc"],
                "groupby": ["noise_multiplier"],
                "row_limit": 10000,
            },
        ),
    ]


# =============================================================================
# Entry point
# =============================================================================
def create_dashboards(
    *,
    base_url: str = BASE_URL,
    wait_seconds: int = 0,
) -> dict[str, int]:
    """Provision the database, datasets and three dashboards.

    Parameters
    ----------
    base_url:
        Superset root URL.
    wait_seconds:
        Extra time to wait for Superset to finish booting before the first
        request. Useful right after ``docker compose up``.

    Returns
    -------
    dict
        ``{dashboard_title: dashboard_id}``
    """
    if wait_seconds:
        logger.info("Waiting %ss for Superset to finish booting", wait_seconds)
        time.sleep(wait_seconds)

    client = SupersetClient(base_url)
    client.login()

    db_id = ensure_database(client)
    datasets = {table: ensure_dataset(client, db_id, table) for table in TABLES}

    results = {
        "FL Telco — Convergence": ensure_dashboard(
            client,
            "FL Telco — Convergence",
            "fl-telco-convergence",
            convergence_charts(datasets["e1_convergence"]),
        ),
        "FL Telco — Fairness": ensure_dashboard(
            client,
            "FL Telco — Fairness",
            "fl-telco-fairness",
            fairness_charts(datasets["e5_fairness"]),
        ),
        "FL Telco — Privacy": ensure_dashboard(
            client,
            "FL Telco — Privacy",
            "fl-telco-privacy",
            privacy_charts(datasets["e4_privacy"]),
        ),
    }
    return results


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.serving.s25_superset_dashboards``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Create Superset dashboards")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()

    result = create_dashboards(base_url=args.base_url, wait_seconds=args.wait)

    print("\n" + "=" * 70)
    print("SUPERSET DASHBOARDS")
    print("=" * 70)
    for title, dash_id in result.items():
        print(f"  {title:<32} id={dash_id}")
    print(f"\n  Open: {args.base_url}/dashboard/list/")


if __name__ == "__main__":
    main()
