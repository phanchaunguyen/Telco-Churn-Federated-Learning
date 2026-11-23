"""Airflow DAG: the full FL Telco pipeline.

Pipeline position: **ORCHESTRATION (wraps every layer)**

This DAG is written to the Airflow 2.x standard but is **not executed** in this
project. Airflow needs a metadata database, a scheduler and a webserver — about
2 GB of containers — and the pipeline is short enough to run by hand. The DAG
exists to show how the stages fit together and to make the orchestration
explicit rather than implied.

Stage order
-----------
``data_prep`` → ``fl_train`` → ``evaluate`` → ``export_dashboards``

Each stage is a separate task so a failure is attributable. Running the whole
pipeline as one shell command would hide which layer broke.

Why ``BashOperator`` and not ``PythonOperator``
-----------------------------------------------
Every stage is already a CLI entry point (``python -m src...``). Wrapping them
in ``BashOperator`` keeps the DAG a thin description of order and dependencies,
with no business logic duplicated into Airflow. The same commands work outside
Airflow, which is how they are actually run in this project.

Usage
-----
>>> from dags.s24_fl_pipeline_dag import dag
>>> dag.dag_id
'fl_telco_pipeline'
"""

from __future__ import annotations

import datetime as dt

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    _HAS_AIRFLOW = True
except ImportError:  # pragma: no cover - Airflow is not installed here
    _HAS_AIRFLOW = False
    DAG = None  # type: ignore[assignment,misc]
    BashOperator = None  # type: ignore[assignment,misc]


# Project root inside the Airflow container. The compose file mounts the repo
# at /opt/airflow/project, so every command runs from there.
PROJECT_DIR = "/opt/airflow/project"

DEFAULT_ARGS = {
    "owner": "fl-telco",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=5),
}


def _build_dag():
    """Construct the DAG. Separated so tests can import it without Airflow."""
    if not _HAS_AIRFLOW:
        raise ImportError(
            "airflow is not installed; this DAG is provided for reference and "
            "is not executed in this project"
        )

    with DAG(
        dag_id="fl_telco_pipeline",
        description="Spark prep -> FL training -> evaluation -> BI export",
        default_args=DEFAULT_ARGS,
        start_date=dt.datetime(2026, 11, 1),
        schedule=None,  # manual trigger only; the pipeline is not periodic
        catchup=False,
        max_active_runs=1,
        tags=["fl", "telco", "capstone"],
    ) as dag:
        # --- Layer 1: data -------------------------------------------------
        data_prep = BashOperator(
            task_id="data_prep",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                "python -m src.data.s01_clean && "
                "python -m src.data.s02_split && "
                "python -m src.data.s03_amplify && "
                "python -m src.data.s04_quality_gate && "
                "python -m src.partitioning.s05_partition"
            ),
            doc_md=(
                "Clean the raw CSV, split real data, amplify to 1M synthetic "
                "rows, run the quality gate, then partition into 10 silos for "
                "each of the three strategies."
            ),
        )

        # --- Layer 2: federated training -----------------------------------
        fl_train = BashOperator(
            task_id="fl_train",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                "python -m src.experiments.s15_e1_fl_gap --fold-ids 0 --rounds 50"
            ),
            doc_md=(
                "Train the federated model for E1 fold 0. This is the run whose "
                "weights are exported for serving."
            ),
        )

        # --- Layer 3: evaluation -------------------------------------------
        evaluate = BashOperator(
            task_id="evaluate",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                "python -m src.experiments.s16_e2_e3 --which both --rounds 50 && "
                "python -m src.experiments.s17_e4_dp --rounds 50 && "
                "python -m src.experiments.s18_e5_fairness --rounds 50 && "
                "python -m src.experiments.s19_e6_scalability --rounds 50 && "
                "python -m src.experiments.s20_e7_communication --rounds 50"
            ),
            doc_md=(
                "Run E2 through E7. Each writes its own JSON and markdown into "
                "results/, which the next task reads."
            ),
        )

        # --- Layer 4: serving and BI ---------------------------------------
        export_dashboards = BashOperator(
            task_id="export_dashboards",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                "python -m src.serving.s21_export_model && "
                "python -m src.serving.s23_export_results"
            ),
            doc_md=(
                "Export the trained model for the API and push the experiment "
                "tables into postgres so Superset and Grafana can read them."
            ),
        )

        data_prep >> fl_train >> evaluate >> export_dashboards

    return dag


if _HAS_AIRFLOW:
    dag = _build_dag()
else:  # pragma: no cover
    dag = None
