#!/bin/bash
# Superset bootstrap for the FL Telco project.
#
# Runs as the container entrypoint. Three steps, in order, each of which must
# succeed before the next:
#
#   1. db upgrade      — create/migrate Superset's own metadata tables
#   2. create-admin    — create the login account (idempotent: skipped if it exists)
#   3. init            — load roles, permissions, default data sources
#
# Then it starts the webserver in the foreground so the container stays alive.
#
# Any failure exits non-zero and the container stops. There is no fallback:
# a half-initialised Superset is worse than a stopped one, because the
# dashboards would silently be missing.

set -euo pipefail

echo "=== [1/4] Upgrading Superset metadata database ==="
superset db upgrade

echo "=== [2/4] Creating admin user ==="
# `superset fab create-admin` fails if the user already exists, which would
# abort the script on every restart. Check first, then create only if absent.
if superset fab list-users 2>/dev/null | grep -q "${ADMIN_USERNAME}"; then
    echo "Admin user '${ADMIN_USERNAME}' already exists, skipping"
else
    superset fab create-admin \
        --username "${ADMIN_USERNAME}" \
        --firstname "${ADMIN_FIRST_NAME}" \
        --lastname "${ADMIN_LAST_NAME}" \
        --email "${ADMIN_EMAIL}" \
        --password "${ADMIN_PASSWORD}"
fi

echo "=== [3/4] Initialising Superset ==="
superset init

echo "=== [4/4] Starting webserver on :8088 ==="
exec superset run -h 0.0.0.0 -p 8088 --with-threads
