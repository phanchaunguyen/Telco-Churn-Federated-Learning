"""Superset configuration for the FL Telco project.

Mounted into the container at ``/app/pythonpath/superset_config.py``. Superset
imports this file at start-up, so anything set here overrides the defaults.

The metadata database is the same postgres instance MLflow uses. That is
deliberate: the experiment tables written by
``python -m src.serving.s23_export_results`` land in the same database, so
Superset can query them without a second connection.
"""

import os

# --- Security ---------------------------------------------------------------
# Overridden by SUPERSET_SECRET_KEY in docker-compose.yml. The fallback exists
# only so the file is importable outside a container.
SECRET_KEY = os.environ.get(
    "SUPERSET_SECRET_KEY", "fltelco-superset-secret-change-me"
)

# --- Metadata database ------------------------------------------------------
SQLALCHEMY_DATABASE_URI = os.environ.get(
    "SUPERSET_DATABASE_URI",
    "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow",
)

# --- Feature flags ----------------------------------------------------------
FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
    "DASHBOARD_NATIVE_FILTERS": True,
    "DASHBOARD_CROSS_FILTERS": True,
}

# --- Behaviour --------------------------------------------------------------
# The dashboards are read-only views over experiment output, so row limits are
# generous and caching is off (the data changes whenever an experiment reruns).
ROW_LIMIT = 10_000
DEFAULT_TIME_GRAIN_SQLA = "P1D"

# Superset warns loudly when this is unset; the project runs on localhost only.
WTF_CSRF_ENABLED = True
TALISMAN_ENABLED = False

# --- Localisation -----------------------------------------------------------
BABEL_DEFAULT_LOCALE = "en"
LANGUAGES = {"en": {"flag": "us", "name": "English"}}
