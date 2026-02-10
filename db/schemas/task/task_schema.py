from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from db.schemas.task.query_schema import QueryCreateRequest
from datetime import datetime

class TaskCreateRequest(BaseModel):
    name: str
    description: str
    metric: str
    pipeline_tags: Optional[List[str]] = None
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
    pipeline_tags: Optional[List[str]] = None

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
    name: str
    split: str
    label: str
    files: List[QueryFileResponse]


class TaskDetailResponse(BaseModel):
    id: str
    name: str
    description: str
    metric: str
    pipeline_tags: Optional[List[str]] = None
    queries: List[QueryDetailResponse]
    created_at: datetime
    updated_at: datetime

class UpdateTaskQueryRequest(BaseModel):
    id: int
    name: str = ""
    split: str
    label: str
    files: list["UpdateTaskFileRequest"]
    existing_files: list[str] = []  # List of object_keys to preserve


class UpdateTaskFileRequest(BaseModel):
    filename: str
    content_type: str


class TaskUpdateRequest(BaseModel):
    name: str
    description: str
    metric: str
    pipeline_tags: Optional[List[str]] = None
    queries: List[UpdateTaskQueryRequest]


class TaskMetadataUpdateRequest(BaseModel):
    """Partial update for task metadata only (no query/file changes)"""
    name: Optional[str] = None
    description: Optional[str] = None
    metric: Optional[str] = None
    pipeline_tags: Optional[List[str]] = None
