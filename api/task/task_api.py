from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from db.session import get_db
from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListResponse,
    TaskDetailResponse,
)
from db.services.task.task_service import (
    create_task,
    list_tasks,
    get_task,
    delete_task,   # ✅ NEW
)
from db.schemas.task.file_schema import CommitFilesRequest
from db.services.task.file_service import commit_files


router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("", response_model=TaskCreateResponse, summary="Create a new task")
def create_task_api(
    payload: TaskCreateRequest,
    db: Session = Depends(get_db),
):
    return create_task(db, payload)


@router.get(
    "",
    response_model=List[TaskListResponse],
    summary="List all tasks",
)
def list_tasks_api(db: Session = Depends(get_db)):
    return list_tasks(db)


@router.get(
    "/{task_id}",
    response_model=TaskDetailResponse,
    summary="Get a task by ID",
)
def get_task_api(
    task_id: str,
    db: Session = Depends(get_db),
):
    try:
        return get_task(db, task_id)
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Task not found")


@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a task",
)
def delete_task_api(
    task_id: str,
    db: Session = Depends(get_db),
):
    try:
        delete_task(db, task_id)
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Task not found")


@router.post(
    "/{task_id}/files/commit",
    summary="Commit uploaded files metadata",
)
def commit_files_api(
    task_id: str,
    payload: CommitFilesRequest,
    db: Session = Depends(get_db),
):
    commit_files(db, task_id, payload)
    return {"status": "ok"}
