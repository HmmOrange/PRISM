"""add updated_at column to tasks

Revision ID: add_updated_at_001
Revises: add_pipeline_tags_001
Create Date: 2026-02-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "add_updated_at_001"
down_revision: Union[str, None] = "add_pipeline_tags_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add updated_at column to tasks table."""
    op.add_column(
        "tasks",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    """Remove updated_at column from tasks table."""
    op.drop_column("tasks", "updated_at")
