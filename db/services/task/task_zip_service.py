import csv
import json
import mimetypes
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from db.models.task.query_file import QueryFileModel
from db.schemas.task.task_schema import TaskDetailResponse
from db.services.task.task_service import get_task, _create_task_and_queries
from storage.storage_factory import get_storage


REQUIRED_ROOT_FILES = {
    "task_description.txt",
    "metadata.json",
}

REQUIRED_SPLITS = {"test", "validation"}


def create_task_from_zip(
    db: Session,
    zip_file: UploadFile,
    user_id: str | None = None,
):
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip")

    tmp_dir = Path(tempfile.mkdtemp(prefix="prism_zip_"))

    try:
        _extract_zip(zip_file, tmp_dir)
        root = _resolve_task_root(tmp_dir)
        _validate_root_structure(root)

        # ✅ USE root, not tmp_dir
        description = _read_description(root)
        metric = _read_metric(root)

        queries, files_by_query = _parse_splits(root)

        task, query_models = _create_task_and_queries(
            db,
            name=zip_file.filename.rsplit(".", 1)[0],
            description=description,
            metric=metric,
            queries=queries,
            user_id=user_id,
        )

        query_id_map = {
            (q.split, q.index): q.id
            for q in query_models
        }

        storage = get_storage()

        for (split, index), files in files_by_query.items():
            query_id = query_id_map[(split, index)]

            for file_path in files:
                # ✅ STANDARDIZE ON inputs/
                object_key = f"{task.id}/{split}/inputs/{index}/{file_path.name}"
                content_type, _ = mimetypes.guess_type(file_path.name)

                storage.upload_file(
                    object_key=object_key,
                    file_path=str(file_path),
                    content_type=content_type,
                )

                db.add(
                    QueryFileModel(
                        query_id=query_id,
                        filename=file_path.name,
                        object_key=object_key,
                        content_type=content_type,
                        size=file_path.stat().st_size,
                    )
                )

        db.commit()
        return get_task(db, str(task.id))

    except HTTPException:
        db.rollback()
        raise

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------- helpers ----------

def _resolve_task_root(tmp_dir: Path) -> Path:
    """
    If the zip extracts into a single top-level directory,
    treat that directory as the task root.
    """
    entries = [p for p in tmp_dir.iterdir() if not p.name.startswith("__")]

    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]

    return tmp_dir


def _extract_zip(zip_file: UploadFile, dest: Path):
    with zipfile.ZipFile(zip_file.file) as z:
        for member in z.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise HTTPException(status_code=400, detail="Unsafe ZIP paths detected")
        z.extractall(dest)


def _validate_root_structure(root: Path):
    for name in REQUIRED_ROOT_FILES:
        if not (root / name).is_file():
            raise HTTPException(status_code=400, detail=f"Missing {name}")

    for split in REQUIRED_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Missing split: {split}")

        if not (split_dir / "labels.csv").is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Missing labels.csv in {split}",
            )

        input_dir = split_dir / "inputs"
        if not input_dir.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Missing inputs/ directory in {split}",
            )


def _read_description(root: Path) -> str:
    return (root / "task_description.txt").read_text(encoding="utf-8").strip()


def _read_metric(root: Path) -> str:
    with open(root / "metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metric" not in data or not isinstance(data["metric"], str):
        raise HTTPException(status_code=400, detail="metadata.json must contain metric")

    return data["metric"]


def _parse_splits(root: Path):
    queries = []
    file_map: dict[tuple[str, int], list[Path]] = {}

    for split in REQUIRED_SPLITS:
        split_dir = root / split
        labels = _read_labels(split_dir / "labels.csv")
        input_dir = split_dir / "inputs"

        for index, label in labels.items():
            query_input_dir = input_dir / str(index)
            if not query_input_dir.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing input folder for {split} index {index}",
                )

            files = [p for p in query_input_dir.iterdir() if p.is_file()]
            if not files:
                raise HTTPException(
                    status_code=400,
                    detail=f"No files found for {split} index {index}",
                )

            queries.append(
                {
                    "index": index,
                    "split": split,
                    "label": label,
                }
            )
            file_map[(split, index)] = files

    return queries, file_map


def _read_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        first_row = next(reader, None)
        if first_row is None:
            raise HTTPException(status_code=400, detail="labels.csv is empty")

        # Detect header row
        try:
            int(first_row[0])
            rows = [first_row]  # first row is data
        except ValueError:
            # first row is header → skip
            rows = []

        rows.extend(reader)

        for row in rows:
            if len(row) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="labels.csv must have index,label columns",
                )

            try:
                index = int(row[0])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="labels.csv index must be integer",
                )

            labels[index] = row[1]

    if not labels:
        raise HTTPException(status_code=400, detail="labels.csv has no valid rows")

    return labels
