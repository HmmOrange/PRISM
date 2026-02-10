"""add pipeline_tags column to tasks

Revision ID: add_pipeline_tags_001
Revises: add_users_001
Create Date: 2026-02-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


# revision identifiers, used by Alembic.
revision: str = "add_pipeline_tags_001"
down_revision: Union[str, None] = "add_users_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add pipeline_tags column to tasks table."""
    op.add_column(
        "tasks",
        sa.Column("pipeline_tags", ARRAY(sa.String()), nullable=True),
    )


def downgrade() -> None:
    """Remove pipeline_tags column from tasks table."""
    op.drop_column("tasks", "pipeline_tags")
