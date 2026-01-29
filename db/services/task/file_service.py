from sqlalchemy.orm import Session

from db.models.task.query import QueryModel
from db.models.task.query_file import QueryFileModel
from db.schemas.task.file_schema import CommitFilesRequest


def commit_files(
    db: Session,
    task_id: str,
    payload: CommitFilesRequest,
):
    for f in payload.files:
        query = (
            db.query(QueryModel)
            .filter(
                QueryModel.task_id == task_id,
                QueryModel.index == f.query_index,
            )
            .first()
        )

        if not query:
            continue  # or raise if you want strictness later

        db.add(
            QueryFileModel(
                query_id=query.id,
                filename=f.filename,
                object_key=f.object_key,
                content_type=f.content_type,
                size=f.size,
            )
        )

    db.commit()
