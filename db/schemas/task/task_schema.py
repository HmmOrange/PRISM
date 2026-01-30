from pydantic import BaseModel
from typing import List, Dict, Any
from db.schemas.task.query_schema import QueryCreateRequest
from datetime import datetime

class TaskCreateRequest(BaseModel):
    name: str
    description: str
    metric: str
    queries: List[QueryCreateRequest]


class PresignedFileResponse(BaseModel):
    filename: str         
    object_key: str        
    url: str                  
    fields: Dict[str, Any] 

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

class QuerySummaryResponse(BaseModel):
    index: int
    split: str
    label: str


class QueryFileResponse(BaseModel):
    filename: str
    object_key: str
    content_type: str
    size: int
    download_url: str


class QueryDetailResponse(BaseModel):
    index: int
    split: str
    label: str
    files: List[QueryFileResponse]


class TaskDetailResponse(BaseModel):
    id: str
    name: str
    description: str
    metric: str
    queries: List[QueryDetailResponse]
