from sqlalchemy.orm import Session
from sqlalchemy import func, case
from sqlalchemy.exc import NoResultFound

from db.models.task.task import TaskModel
from db.models.task.query import QueryModel
from db.models.task.query_file import QueryFileModel

from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
    QueryUploadResponse,
    PresignedFileResponse,
    TaskDetailResponse,
    QueryDetailResponse,
    QueryFileResponse,
    TaskListResponse,
)

from storage.storage_factory import get_storage


def create_task(
    db: Session,
    payload: TaskCreateRequest,
    user_id: str | None = None,
) -> TaskCreateResponse:
    storage = get_storage()

    task = TaskModel(
        name=payload.name,
        description=payload.description,
        metric=payload.metric,
        user_id=user_id,
    )
    db.add(task)
    db.flush()

    for q in payload.queries:
        db.add(
            QueryModel(
                task_id=task.id,
                index=q.id,
                name=getattr(q, 'name', '') or "",
                split=q.split,
                label=q.label or "",
            )
        )

    db.commit()
    db.refresh(task)

    uploads: list[QueryUploadResponse] = []

    for q in payload.queries:
        files: list[PresignedFileResponse] = []

        for f in q.files:
            object_key = f"{task.id}/{q.split}/input/{q.id}/{f.filename}"

            post = storage.generate_presigned_upload_post(
                object_key=object_key,
                content_type=f.content_type,
            )

            files.append(
                PresignedFileResponse(
                    filename=f.filename,
                    object_key=object_key,
                    url=post["url"],
                    fields=post["fields"],
                )
            )


        uploads.append(
            QueryUploadResponse(
                query_index=q.id,
                files=files,
            )
        )

    return TaskCreateResponse(
        task_id=str(task.id),
        uploads=uploads,
    )


def list_tasks(db: Session, user_id: str | None = None) -> list[TaskListResponse]:
    query = (
        db.query(
            TaskModel.id,
            TaskModel.name,
            TaskModel.description,
            TaskModel.metric,
            TaskModel.created_at,
            func.count(QueryModel.id).label("total_queries"),
            func.sum(
                case((QueryModel.split == "test", 1), else_=0)
            ).label("test_queries"),
            func.sum(
                case((QueryModel.split == "validation", 1), else_=0)
            ).label("validation_queries"),
        )
        .outerjoin(QueryModel, QueryModel.task_id == TaskModel.id)
    )

    if user_id is not None:
        query = query.filter(TaskModel.user_id == user_id)

    rows = (
        query
        .group_by(TaskModel.id)
        .order_by(TaskModel.created_at.desc())
        .all()
    )

    return [
        TaskListResponse(
            id=str(r.id),
            name=r.name,
            description=r.description,
            metric=r.metric,
            total_queries=r.total_queries or 0,
            test_queries=r.test_queries or 0,
            validation_queries=r.validation_queries or 0,
            created_at=r.created_at,
        )
        for r in rows
    ]


def get_task(db: Session, task_id: str, user_id: str | None = None) -> TaskDetailResponse:
    query = db.query(TaskModel).filter(TaskModel.id == task_id)
    if user_id is not None:
        query = query.filter(TaskModel.user_id == user_id)
    task = query.first()
    if not task:
        raise NoResultFound()

    queries = (
        db.query(QueryModel)
        .filter(QueryModel.task_id == task.id)
        .order_by(QueryModel.index)
        .all()
    )

    query_ids = [q.id for q in queries]

    files = (
        db.query(QueryFileModel)
        .filter(QueryFileModel.query_id.in_(query_ids))
        .all()
    )

    files_by_query: dict[int, list[QueryFileModel]] = {}
    for f in files:
        files_by_query.setdefault(f.query_id, []).append(f)

    return TaskDetailResponse(
        id=str(task.id),
        name=task.name,
        description=task.description,
        metric=task.metric,
        queries=[
            QueryDetailResponse(
                index=q.index,
                name=q.name or "",
                split=q.split,
                label=q.label,
                files=[
                    QueryFileResponse(
                        filename=f.filename,
                        object_key=f.object_key,
                        content_type=f.content_type,
                        size=f.size,
                        # 🔑 backend proxy, NOT presigned
                        download_url=f"/storage/download?object_key={f.object_key}",
                    )
                    for f in files_by_query.get(q.id, [])
                ],
            )
            for q in queries
        ],
    )

def delete_task(db: Session, task_id: str, user_id: str | None = None) -> None:
    query = db.query(TaskModel).filter(TaskModel.id == task_id)
    if user_id is not None:
        query = query.filter(TaskModel.user_id == user_id)
    task = query.first()
    if not task:
        raise NoResultFound()

    # Delete files → queries → task
    query_ids = (
        db.query(QueryModel.id)
        .filter(QueryModel.task_id == task.id)
        .subquery()
    )

    db.query(QueryFileModel).filter(
        QueryFileModel.query_id.in_(query_ids)
    ).delete(synchronize_session=False)

    db.query(QueryModel).filter(
        QueryModel.task_id == task.id
    ).delete(synchronize_session=False)

    db.delete(task)
    db.commit()

def _create_task_and_queries(
    db: Session,
    *,
    name: str,
    description: str,
    metric: str,
    queries: list[dict],
    user_id: str | None = None,
):
    """
    Creates TaskModel and QueryModel rows.
    Returns:
        task: TaskModel
        query_models: list[QueryModel] (IDs are populated)
    """
    task = TaskModel(
        name=name,
        description=description,
        metric=metric,
        user_id=user_id,
    )
    db.add(task)
    db.flush()  # assigns task.id

    query_models: list[QueryModel] = []

    for q in queries:
        qm = QueryModel(
            task_id=task.id,
            index=q["index"],
            name=q.get("name", ""),
            split=q["split"],
            label=q.get("label", ""),
        )
        db.add(qm)
        query_models.append(qm)

    db.flush()  # assigns QueryModel.id

    return task, query_models

def update_task(
    db: Session,
    task_id: str,
    payload,
    user_id: str | None = None,
) -> TaskCreateResponse:
    storage = get_storage()

    query = db.query(TaskModel).filter(TaskModel.id == task_id)
    if user_id is not None:
        query = query.filter(TaskModel.user_id == user_id)
    task = query.first()
    if not task:
        raise NoResultFound()

    # ---- Update task metadata ----
    task.name = payload.name
    task.description = payload.description
    task.metric = payload.metric

    # ---- Delete existing queries + files ----
    query_ids = (
        db.query(QueryModel.id)
        .filter(QueryModel.task_id == task.id)
        .subquery()
    )

    db.query(QueryFileModel).filter(
        QueryFileModel.query_id.in_(query_ids)
    ).delete(synchronize_session=False)

    db.query(QueryModel).filter(
        QueryModel.task_id == task.id
    ).delete(synchronize_session=False)

    db.flush()

    # ---- Recreate queries ----
    for q in payload.queries:
        db.add(
            QueryModel(
                task_id=task.id,
                index=q.id,
                name=getattr(q, 'name', '') or "",
                split=q.split,
                label=q.label or "",
            )
        )

    db.commit()
    db.refresh(task)

    # ---- Generate presigned uploads (same as create) ----
    uploads: list[QueryUploadResponse] = []

    for q in payload.queries:
        files: list[PresignedFileResponse] = []

        for f in q.files:
            object_key = f"{task.id}/{q.split}/input/{q.id}/{f.filename}"

            post = storage.generate_presigned_upload_post(
                object_key=object_key,
                content_type=f.content_type,
            )

            files.append(
                PresignedFileResponse(
                    filename=f.filename,
                    object_key=object_key,
                    url=post["url"],
                    fields=post["fields"],
                )
            )

        uploads.append(
            QueryUploadResponse(
                query_index=q.id,
                files=files,
            )
        )

    return TaskCreateResponse(
        task_id=str(task.id),
        uploads=uploads,
    )
