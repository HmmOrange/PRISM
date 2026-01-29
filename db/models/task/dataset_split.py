import enum
import uuid
from sqlalchemy import Column, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from db.base import Base


class SplitType(enum.Enum):
    test = "test"
    validation = "validation"


class DatasetSplitModel(Base):
    __tablename__ = "dataset_splits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    split_type = Column(Enum(SplitType), nullable=False)
    labels_uri = Column(String, nullable=False)
