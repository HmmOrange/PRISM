#!/bin/sh
set -e

echo "Starting PRISM Tasks Server..."

exec uvicorn server.tasks_server:app \
  --host 0.0.0.0 \
  --port 8000
