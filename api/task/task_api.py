
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
)
from db.schemas.task.task_schema import TaskListResponse
from db.services.task.task_service import create_task
from db.services.task.task_service import list_tasks


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
