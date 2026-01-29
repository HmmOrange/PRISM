from pydantic import BaseModel, Field
from typing import List


class QueryFileRequest(BaseModel):
    filename: str
    content_type: str


class QueryCreateRequest(BaseModel):
    id: int = Field(ge=0)
    split: str
    label: str = ""
    files: List[QueryFileRequest]
