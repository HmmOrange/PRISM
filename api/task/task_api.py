from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.session import get_db
from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
)

from db.services.task.task_service import create_task

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("", response_model=TaskCreateResponse)
def create_task_api(
    payload: TaskCreateRequest,
    db: Session = Depends(get_db),
):
    return create_task(db, payload)
