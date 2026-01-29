from pydantic import BaseModel
from typing import List
from uuid import UUID


class FileCommitRequest(BaseModel):
    query_index: int
    filename: str
    object_key: str
    content_type: str
    size: int


class CommitFilesRequest(BaseModel):
    files: List[FileCommitRequest]
