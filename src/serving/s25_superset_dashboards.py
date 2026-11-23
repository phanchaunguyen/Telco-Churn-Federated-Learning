"""Create the three Superset dashboards via the Superset REST API.

Pipeline position: **EXPORT (BI) → DASHBOARD**

Superset stores dashboards as rows in its metadata database, not as files, so
they cannot be provisioned by mounting a JSON file the way Grafana can. This
script logs in, registers the postgres connection, registers the three tables
as datasets, and creates one dashboard per experiment theme.

What the API actually requires
------------------------------
Four things are easy to get wrong and each fails with a different error:

1. **``query_context`` is mandatory.** A chart created with only ``params``
   returns 400 "Chart has no query context saved" when queried.
2. **Metrics must be objects, not strings.** ``"metrics": ["auc"]`` gives
   "Metric 'auc' does not exist"; the full
   ``{expressionType, column, aggregate, label}`` form is required.
3. **``orderby`` is a list of ``[column, ascending]`` pairs**, not a list of
   dicts. A dict gives "Not a tuple".
4. **``position_json`` needs the full grid skeleton** (``DASHBOARD_VERSION_KEY``,
   ``ROOT_ID``, ``GRID_ID``, ``parents``). Without it the dashboard renders as
   a permanent "Loading" spinner.

Idempotent
----------
Charts are deleted and recreated on every run so a changed SQL takes effect.
The dashboard itself is reused if it already exists.

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
            json=payload,
            timeout=60,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"POST {path} failed ({response.status_code}): {response.text}"
            )
        return response.json()

    # ------------------------------------------------------------------
    def delete(self, path: str) -> None:
        response = self.session.delete(
            f"{self.base_url}{path}", headers=self._headers(), timeout=30
        )
        if response.status_code not in (200, 204):
            raise RuntimeError(
                f"DELETE {path} failed ({response.status_code}): {response.text}"
            )


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


# =============================================================================
# Chart construction
# =============================================================================
def _metric(column: str, aggregate: str = "AVG") -> dict[str, Any]:
    """Build a metric object. A bare string is rejected by the API."""
    return {
        "expressionType": "SIMPLE",
        "column": {"column_name": column},
        "aggregate": aggregate,
        "label": f"{aggregate}({column})",
    }


def _filter(column: str, op: str, value: Any) -> dict[str, Any]:
    return {"col": column, "op": op, "val": value, "expressionType": "SIMPLE"}


def _chart_payload(
    *,
    name: str,
    dataset_id: int,
    viz_type: str,
    x_axis: str | None,
    metric_column: str,
    groupby: list[str],
    filters: list[dict[str, Any]],
    orderby: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Build a complete chart payload including ``query_context``.

    ``query_context`` is what the chart data endpoint reads. Without it the
    chart exists but every query returns 400.
    """
    metric = _metric(metric_column)
    columns = [x_axis] if x_axis else []

    params: dict[str, Any] = {
        "datasource": f"{dataset_id}__table",
        "viz_type": viz_type,
        "metrics": [metric],
        "groupby": groupby,
        "adhoc_filters": filters,
        "row_limit": 10_000,
    }
    if x_axis:
        params["x_axis"] = x_axis

    query: dict[str, Any] = {
        "columns": columns + groupby,
        "metrics": [metric],
        "filters": [
            {"col": f["col"], "op": f["op"], "val": f["val"]} for f in filters
        ],
        "row_limit": 10_000,
        "orderby": orderby or [],
        "extras": {"having": "", "where": ""},
        "is_timeseries": False,
        "granularity": None,
    }
    query_context = {
        "datasource": {"id": dataset_id, "type": "table"},
        "force": False,
        "queries": [query],
        "form_data": params,
        "result_format": "json",
        "result_type": "full",
    }

    return {
        "slice_name": name,
        "viz_type": viz_type,
        "datasource_id": dataset_id,
        "datasource_type": "table",
        "params": json.dumps(params),
        "query_context": json.dumps(query_context),
    }


def _delete_charts_by_name(client: SupersetClient, names: set[str]) -> None:
    """Remove charts with these names so a rerun does not duplicate them."""
    listing = client.get("/api/v1/chart/", q=json.dumps({"filters": []}))
    for chart in listing.get("result", []):
        if chart["slice_name"] in names:
            client.delete(f"/api/v1/chart/{chart['id']}")
            logger.info("  removed stale chart %r", chart["slice_name"])


def _position_json(chart_ids: list[int]) -> str:
    """Build a valid dashboard layout.

    Superset needs the full grid skeleton. A bare ``{CHART-id: {...}}`` map
    leaves the dashboard stuck on "Loading" because the renderer cannot find
    ``ROOT_ID`` or the grid it should attach to.
    """
    position: dict[str, Any] = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "GRID_ID": {
            "type": "GRID",
            "id": "GRID_ID",
            "children": [f"ROW-{cid}" for cid in chart_ids],
            "parents": ["ROOT_ID"],
        },
        "HEADER_ID": {"type": "HEADER", "id": "HEADER_ID", "meta": {"text": ""}},
    }
    for cid in chart_ids:
        row_id = f"ROW-{cid}"
        chart_id = f"CHART-{cid}"
        position[row_id] = {
            "type": "ROW",
            "id": row_id,
            "children": [chart_id],
            "parents": ["ROOT_ID", "GRID_ID"],
            "meta": {"background": "BACKGROUND_TRANSPARENT"},
        }
        position[chart_id] = {
            "type": "CHART",
            "id": chart_id,
            "children": [],
            "parents": ["ROOT_ID", "GRID_ID", row_id],
            "meta": {"width": 6, "height": 50, "chartId": cid},
        }
    return json.dumps(position)


def ensure_dashboard(
    client: SupersetClient,
    title: str,
    slug: str,
    charts: list[dict[str, Any]],
) -> int:
    """Create a dashboard with the given charts, returning its id.

    Charts are recreated on every run so a changed SQL takes effect. The
    dashboard is deleted and recreated too, because Superset 3.1's REST API
    does not write the ``dashboard_slices`` join table: a dashboard created
    through the API has an empty chart list and renders as a permanent
    "Loading" spinner. Recreating it here and linking the charts afterwards
    (see ``link_charts``) is the only reliable path.
    """
    _delete_charts_by_name(client, {c["slice_name"] for c in charts})

    chart_ids: list[int] = []
    for spec in charts:
        created = client.post("/api/v1/chart/", spec)
        chart_ids.append(int(created["id"]))
        logger.info("  chart %r -> id=%s", spec["slice_name"], created["id"])

    # Remove any previous dashboard with this title so the layout is rebuilt
    # against the chart ids that exist now.
    existing = client.get(
        "/api/v1/dashboard/",
        q=json.dumps({"filters": [{"col": "dashboard_title", "opr": "eq", "value": title}]}),
    )
    for dash in existing.get("result", []):
        if dash["dashboard_title"] == title:
            client.delete(f"/api/v1/dashboard/{dash['id']}")
            logger.info("  removed stale dashboard %r", title)

    created = client.post(
        "/api/v1/dashboard/",
        {
            "dashboard_title": title,
            "slug": slug,
            "published": True,
            "position_json": _position_json(chart_ids),
        },
    )
    dash_id = int(created["id"])
    logger.info("Created dashboard %r (id=%s)", title, dash_id)
    return dash_id


def link_charts(dashboard_id: int, chart_ids: list[int], dsn: str | None = None) -> None:
    """Write the ``dashboard_slices`` rows the REST API leaves empty.

    Superset 3.1 exposes no API field for this join table (``slices`` is
    rejected as an unknown field on both POST and PUT). Without these rows the
    dashboard has no charts and the UI shows "There is no chart definition
    associated with this component".

    This is a direct database write, which is why it is isolated in its own
    function with an explicit comment: it is the one place the script bypasses
    the API, and it should be revisited if a later Superset version adds the
    field.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sqlalchemy is required to link charts; install with "
            "`pip install sqlalchemy psycopg2-binary`"
        ) from exc

    engine = create_engine(dsn or DB_URI.replace("postgres:5432", "localhost:5432"))
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM dashboard_slices WHERE dashboard_id = :d"),
            {"d": dashboard_id},
        )
        for cid in chart_ids:
            conn.execute(
                text(
                    "INSERT INTO dashboard_slices (dashboard_id, slice_id) "
                    "VALUES (:d, :s)"
                ),
                {"d": dashboard_id, "s": cid},
            )
    logger.info("  linked %s charts to dashboard %s", len(chart_ids), dashboard_id)


# =============================================================================
# Chart specs per dashboard
# =============================================================================
def convergence_charts(conv_id: int) -> list[dict[str, Any]]:
    """AUC per round for the three E2 partitions."""
    return [
        _chart_payload(
            name="AUC per round — IID",
            dataset_id=conv_id,
            viz_type="echarts_timeseries_line",
            x_axis="round",
            metric_column="auc",
            groupby=[],
            filters=[_filter("run", "==", "iid")],
            orderby=[["round", True]],
        ),
        _chart_payload(
            name="AUC per round — non-IID",
            dataset_id=conv_id,
            viz_type="echarts_timeseries_line",
            x_axis="round",
            metric_column="auc",
            groupby=["run"],
            filters=[_filter("run", "!=", "iid")],
            orderby=[["round", True]],
        ),
    ]


def fairness_charts(fair_id: int) -> list[dict[str, Any]]:
    """Per-client AUC and per-reason recall, FedAvg against FedProx."""
    return [
        _chart_payload(
            name="Per-client AUC",
            dataset_id=fair_id,
            viz_type="echarts_timeseries_bar",
            x_axis="label",
            metric_column="value",
            groupby=["strategy"],
            filters=[_filter("view", "==", "client")],
        ),
        _chart_payload(
            name="Recall by churn reason",
            dataset_id=fair_id,
            viz_type="echarts_timeseries_bar",
            x_axis="label",
            metric_column="value",
            groupby=["strategy"],
            filters=[_filter("view", "==", "reason")],
        ),
    ]


def privacy_charts(priv_id: int) -> list[dict[str, Any]]:
    """AUC against epsilon."""
    return [
        _chart_payload(
            name="Privacy-utility curve",
            dataset_id=priv_id,
            viz_type="echarts_timeseries_line",
            x_axis="epsilon",
            metric_column="auc",
            groupby=[],
            filters=[],
            orderby=[["epsilon", True]],
        ),
        _chart_payload(
            name="AUC by noise multiplier",
            dataset_id=priv_id,
            viz_type="echarts_timeseries_bar",
            x_axis="noise_multiplier",
            metric_column="auc",
            groupby=[],
            filters=[],
            orderby=[["noise_multiplier", True]],
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

    specs = {
        "FL Telco — Convergence": (
            "fl-telco-convergence",
            convergence_charts(datasets["e1_convergence"]),
        ),
        "FL Telco — Fairness": (
            "fl-telco-fairness",
            fairness_charts(datasets["e5_fairness"]),
        ),
        "FL Telco — Privacy": (
            "fl-telco-privacy",
            privacy_charts(datasets["e4_privacy"]),
        ),
    }

    results: dict[str, int] = {}
    for title, (slug, charts) in specs.items():
        dash_id = ensure_dashboard(client, title, slug, charts)
        # The API cannot write dashboard_slices, so link the charts directly.
        chart_ids = [
            int(c["id"])
            for c in client.get(
                "/api/v1/chart/", q=json.dumps({"filters": []})
            ).get("result", [])
            if c["slice_name"] in {spec["slice_name"] for spec in charts}
        ]
        link_charts(dash_id, chart_ids)
        results[title] = dash_id

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
