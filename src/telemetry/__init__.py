"""Kafka telemetry — observability bus for the federated loop.

Kafka carries **metrics only**. Model weights travel over gRPC/Flower and
nothing else. That separation is the core architectural constraint of the
project: if weights ever moved through Kafka, the FL protocol would no longer
be standard FL.

Modules
-------
* ``s29_producer`` — publishes per-client and per-round metrics
* ``s30_consumer`` — drains both topics into postgres for the dashboards
"""
