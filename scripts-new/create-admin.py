#!/usr/bin/env python3
"""Bootstrap script — idempotently creates the ``admin`` superuser.

Implements SRS 4.1 Admin Seed Script Specifications:
1. Check if a user with username ``admin`` exists.
2. If yes, log "Admin exists" and exit.
3. If no, hash the password using bcrypt and insert the record.

Usage (local with Docker database):
    DB_HOST=localhost python scripts-new/create-admin.py

Usage (inside Docker container):
    docker exec prism_tasks_server python scripts-new/create-admin.py

Usage (via docker compose):
    docker compose exec tasks-server python scripts-new/create-admin.py

Environment Variables:
    DB_HOST     - Database host (default: from config.yaml, typically 'postgres')
    DB_PORT     - Database port (default: 5432)
    DB_NAME     - Database name (default: 'prism')
    DB_USER     - Database user (default: 'prism')
    DB_PASSWORD - Database password (default: 'prism')
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports resolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db.session import SessionLocal
from db.models.user.user import UserModel
from db.services.user.auth_service import hash_password, get_user_by_username


ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin@space"
ADMIN_EMAIL = "admin@prism.local"


def main() -> None:
    db = SessionLocal()
    try:
        existing = get_user_by_username(db, ADMIN_USERNAME)
        if existing:
            print(f"Admin exists (id={existing.id})")
            return

        user = UserModel(
            username=ADMIN_USERNAME,
            email=ADMIN_EMAIL,
            hashed_password=hash_password(ADMIN_PASSWORD),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Admin created (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
