from sqlalchemy.orm import Session

from db.models.task.task import TaskModel
from db.models.task.query import QueryModel

from db.schemas.task.task_schema import (
    TaskCreateRequest,
    TaskCreateResponse,
    QueryUploadResponse,
    PresignedFileResponse,
)

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
