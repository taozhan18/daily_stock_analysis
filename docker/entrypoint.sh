#!/bin/bash
set -e

# Fix volume permissions when mounted by Docker (root-owned by default)
if [ "$(id -u)" = "0" ]; then
    chown -R dsa:dsa /app/data /app/logs /app/reports 2>/dev/null || true
    exec gosu dsa "$@"
fi

exec "$@"
