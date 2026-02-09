from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound

from db.models.task.task import TaskModel
from db.models.task.query import QueryModel
from db.models.task.query_file import QueryFileModel
from db.schemas.task.file_schema import CommitFilesRequest


def commit_files(
    db: Session,
    task_id: str,
    payload: CommitFilesRequest,
    user_id: str | None = None,
):
    # Verify task exists and belongs to user
    query = db.query(TaskModel).filter(TaskModel.id == task_id)
    if user_id is not None:
        query = query.filter(TaskModel.user_id == user_id)
    task = query.first()
    if not task:
        raise NoResultFound()

    for f in payload.files:
        q = (
            db.query(QueryModel)
            .filter(
                QueryModel.task_id == task_id,
                QueryModel.index == f.query_index,
            )
            .first()
        )

        if not q:
            continue  # or raise if you want strictness later

        db.add(
            QueryFileModel(
                query_id=q.id,
                filename=f.filename,
                object_key=f.object_key,
                content_type=f.content_type,
                size=f.size,
            )
        )

    db.commit()
