"""Layer 4 — serving and BI.

Modules are numbered by pipeline order:

* ``s21_export_model`` — train the E1 fold-0 configuration and persist it
* ``s22_api`` — FastAPI prediction endpoint
* ``s23_export_results`` — push experiment results into postgres for Superset
"""
