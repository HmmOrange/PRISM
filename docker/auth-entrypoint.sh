#!/bin/sh
set -e

echo "Waiting for PostgreSQL to be ready..."
max_attempts=30
attempt=0
while ! nc -z postgres 5432 2>/dev/null; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "PostgreSQL not available after $max_attempts attempts, proceeding anyway..."
        break
    fi
    echo "Waiting for PostgreSQL... attempt $attempt/$max_attempts"
    sleep 2
done

echo "Running database migrations..."
alembic upgrade head

echo "Creating default admin account..."
python scripts-new/create-admin.py

echo "Auth initialization complete."
