import uuid
from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    BigInteger,
    Integer,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from db.base import Base


class QueryFileModel(Base):
    __tablename__ = "query_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    query_id = Column(
        Integer,
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
    )

    filename = Column(String, nullable=False)
    object_key = Column(String, nullable=False, unique=True)
    content_type = Column(String, nullable=False)
    size = Column(BigInteger, nullable=False)

    query = relationship("QueryModel", back_populates="files")
