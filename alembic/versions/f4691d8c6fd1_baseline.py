"""baseline

Revision ID: f4691d8c6fd1
Revises: 
Create Date: 2026-02-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "f4691d8c6fd1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables: tasks, queries, query_files."""

    # 1. Tasks table
    op.create_table(
        "tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("metric", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # 2. Queries table
    op.create_table(
        "queries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("index", sa.Integer, nullable=False),
        sa.Column("task_id", UUID(as_uuid=True), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("split", sa.String(32), nullable=False),
        sa.Column("label", sa.String, nullable=False, server_default=""),
    )
    op.create_index("ix_queries_task_id", "queries", ["task_id"])

    # 3. Query files table
    op.create_table(
        "query_files",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", sa.Integer, sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String, nullable=False),
        sa.Column("object_key", sa.String, nullable=False, unique=True),
        sa.Column("content_type", sa.String, nullable=False),
        sa.Column("size", sa.BigInteger, nullable=False),
    )
    op.create_index("ix_query_files_query_id", "query_files", ["query_id"])


def downgrade() -> None:
    """Drop all baseline tables."""
    op.drop_index("ix_query_files_query_id", table_name="query_files")
    op.drop_table("query_files")
    op.drop_index("ix_queries_task_id", table_name="queries")
    op.drop_table("queries")
    op.drop_table("tasks")
