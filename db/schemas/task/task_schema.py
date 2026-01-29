from pydantic import BaseModel
from typing import List
from db.schemas.task.query_schema import QueryCreateRequest

class TaskCreateRequest(BaseModel):
    name: str
    description: str
    metric: str
    queries: List[QueryCreateRequest]


class PresignedFileResponse(BaseModel):
    filename: str
    upload_url: str
    object_key: str


class QueryUploadResponse(BaseModel):
    query_index: int
    files: List[PresignedFileResponse]


class TaskCreateResponse(BaseModel):
    task_id: str
    uploads: List[QueryUploadResponse]
