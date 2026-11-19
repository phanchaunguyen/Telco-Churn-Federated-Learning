#!/usr/bin/env bash
# =============================================================================
# Bring up the local service stack (MLflow + MinIO + Postgres).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Starting services..."
docker compose up -d

echo
echo "Waiting for health checks..."
sleep 5
docker compose ps

echo
echo "  MLflow UI      : http://localhost:5000"
echo "  MinIO console  : http://localhost:9001  (minioadmin / minioadmin)"
echo "  MinIO S3 API   : http://localhost:9000"
