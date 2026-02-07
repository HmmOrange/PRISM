"""add_name_column_to_queries

Revision ID: add_query_name_001
Revises: 
Create Date: 2026-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_query_name_001'
down_revision: Union[str, None] = 'f4691d8c6fd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add name column to queries table."""
    op.add_column(
        'queries',
        sa.Column('name', sa.String(256), nullable=False, server_default='')
    )


def downgrade() -> None:
    """Remove name column from queries table."""
    op.drop_column('queries', 'name')
