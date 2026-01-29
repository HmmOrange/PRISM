from pydantic import BaseModel
from typing import List
from db.schemas.task.query_schema import QueryCreateRequest
from datetime import datetime

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

class TaskListResponse(BaseModel):
    id: str
    name: str
    description: str
    metric: str

    total_queries: int
    test_queries: int
    validation_queries: int

    created_at: datetime