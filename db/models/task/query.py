from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base


class QueryModel(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True)

    # 0..N-1 index (frontend contract)
    index = Column(Integer, nullable=False)

    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    split = Column(String(32), nullable=False)  # "test" | "validation"
    label = Column(String, nullable=False, default="")

    # Relationships
    task = relationship("TaskModel", back_populates="queries")
