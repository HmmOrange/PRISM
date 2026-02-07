"""baseline

Revision ID: f4691d8c6fd1
Revises: 
Create Date: 2026-02-07

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4691d8c6fd1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Baseline migration (no-op)."""
    pass


def downgrade() -> None:
    """Baseline downgrade (no-op)."""
    pass
