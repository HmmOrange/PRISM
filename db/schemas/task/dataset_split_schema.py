from pydantic import BaseModel
from uuid import UUID
from typing import Literal


class DatasetSplitResponse(BaseModel):
    id: UUID
    task_id: UUID
    split_type: Literal["test", "validation"]
    labels_uri: str

    class Config:
        from_attributes = True
