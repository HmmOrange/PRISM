import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from db.base import Base

# Import ALL models so Alembic sees them
from db.base import Base

# Import model packages so all models register with Base.metadata
import db.models.task  # noqa: F401

# =========================
# Path setup
# =========================
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

# =========================
# Alembic config
# =========================
config = context.config

from utils.constants import DATABASE_URL
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Logging
fileConfig(config.config_file_name)

# =========================
# Metadata
# =========================
target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
