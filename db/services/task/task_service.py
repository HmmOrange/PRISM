from sqlalchemy.orm import Session
from sqlalchemy import func, case
from sqlalchemy.exc import NoResultFound
from db.models.task.task import TaskModel
from db.models.task.query import QueryModel
from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
    QueryUploadResponse,
    PresignedFileResponse,
    TaskDetailResponse,
    QuerySummaryResponse,
    QueryDetailResponse,
    QueryFileResponse,
)
from db.models.task.query_file import QueryFileModel

from db.schemas.task.task_schema import TaskListResponse
from storage.storage_factory import get_storage

def create_task(
    db: Session,
    payload: TaskCreateRequest,
) -> TaskCreateResponse:
    storage = get_storage()

    # 1️⃣ Create task
    task = TaskModel(
        name=payload.name,
        description=payload.description,
        metric=payload.metric,
    )
    db.add(task)
    db.flush()  # ensures task.id is available

    # 2️⃣ Create queries (DB state only, ONCE)
    for q in payload.queries:
        query = QueryModel(
            task_id=task.id,
            index=q.id,          # 0..N-1 (frontend enforces ordering)
            split=q.split,
            label=q.label or "",
        )
        db.add(query)

    # 3️⃣ Commit DB state
    db.commit()
    db.refresh(task)

    # 4️⃣ Generate presigned upload URLs (NO DB WRITES)
    uploads: list[QueryUploadResponse] = []

    for q in payload.queries:
        files: list[PresignedFileResponse] = []

        for f in q.files:
            object_key = (
                f"tasks/{task.id}/"
                f"{q.split}/input/{q.id}/{f.filename}"
            )

            upload_url = storage.generate_presigned_upload_url(
                object_key=object_key,
            )

            files.append(
                PresignedFileResponse(
                    filename=f.filename,
                    upload_url=upload_url,
                    object_key=object_key,
                )
            )

        uploads.append(
            QueryUploadResponse(
                query_index=q.id,
                files=files,
            )
        )

    # 5️⃣ Return response
    return TaskCreateResponse(
        task_id=str(task.id),
        uploads=uploads,
    )

def list_tasks(db: Session) -> list[TaskListResponse]:
    """
    Return all tasks with aggregated query counts.
    """

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
        return None

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

    files_by_query = {}
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
                    )
                    for f in files_by_query.get(q.id, [])
                ],
            )
            for q in queries
        ],
    )