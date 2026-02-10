from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import UploadFile, File, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from db.session import get_db
from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListResponse,
    TaskDetailResponse,
    TaskUpdateRequest,
    TaskMetadataUpdateRequest,
)
from db.services.task.task_service import (
    create_task,
    list_tasks,
    get_task,
    delete_task,
    update_task,
    update_task_metadata,
)
from db.services.task.task_zip_service import create_task_from_zip
from db.schemas.task.file_schema import CommitFilesRequest
from db.services.task.file_service import commit_files
from api.deps import get_current_user
from db.models.user.user import UserModel


router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("", response_model=TaskCreateResponse, summary="Create a new task")
def create_task_api(
    payload: TaskCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    return create_task(db, payload, user_id=str(current_user.id))


@router.get(
    "",
    response_model=List[TaskListResponse],
    summary="List tasks for the current user",
)
def list_tasks_api(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    return list_tasks(db, user_id=str(current_user.id))


@router.get(
    "/{task_id}",
    response_model=TaskDetailResponse,
    summary="Get a task by ID",
)
def get_task_api(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return get_task(db, task_id, user_id=str(current_user.id))
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
    current_user: UserModel = Depends(get_current_user),
):
    try:
        delete_task(db, task_id, user_id=str(current_user.id))
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
    current_user: UserModel = Depends(get_current_user),
):
    try:
        commit_files(db, task_id, payload, user_id=str(current_user.id))
        return {"status": "ok"}
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Task not found")

@router.post(
    "/import/zip",
    response_model=TaskDetailResponse,
    summary="Create task from ZIP archive",
)
def import_task_from_zip_api(
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    return create_task_from_zip(db, zip_file, user_id=str(current_user.id))

@router.put(
    "/{task_id}",
    response_model=TaskCreateResponse,
    summary="Update a task",
)
def update_task_api(
    task_id: str,
    payload: TaskUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return update_task(db, task_id, payload, user_id=str(current_user.id))
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Task not found")


@router.patch(
    "/{task_id}",
    response_model=TaskDetailResponse,
    summary="Partially update task metadata only (name, description, metric, pipeline_tags)",
)
def patch_task_metadata_api(
    task_id: str,
    payload: TaskMetadataUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    try:
        return update_task_metadata(db, task_id, payload, user_id=str(current_user.id))
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Task not found")
