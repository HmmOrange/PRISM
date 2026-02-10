import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from db.base import Base


class TaskModel(Base):
    __tablename__ = "tasks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    metric = Column(String(128), nullable=False)
    pipeline_tags = Column(ARRAY(String), nullable=True, default=[])

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    queries = relationship(
        "QueryModel",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="QueryModel.index",
    )
    owner = relationship("UserModel", backref="tasks")
