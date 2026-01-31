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
) -> TaskCreateResponse:
    storage = get_storage()

    task = TaskModel(
        name=payload.name,
        description=payload.description,
        metric=payload.metric,
    )
    db.add(task)
    db.flush()

    for q in payload.queries:
        db.add(
            QueryModel(
                task_id=task.id,
                index=q.id,
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


def list_tasks(db: Session) -> list[TaskListResponse]:
    rows = (
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


def get_task(db: Session, task_id: str) -> TaskDetailResponse:
    task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
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

def delete_task(db: Session, task_id: str) -> None:
    task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
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