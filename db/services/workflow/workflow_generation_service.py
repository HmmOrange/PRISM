import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List
import os
import logging

from sqlalchemy.orm import Session

from db.models.task.task import TaskModel
from db.models.task.query import QueryModel
from db.models.task.query_file import QueryFileModel
from storage.storage_factory import get_storage

from scripts.operations import generate_and_run

logger = logging.getLogger("workflow_generation")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def start_workflow_generation_job(
    *,
    db: Session,
    task_ids: List[str],
    pipeline_path: str,
    rounds: int,
    run_after: bool = False,
    calculate_after: bool = False,
) -> str:
    job_id = str(uuid.uuid4())

    # Create results dir immediately so status can be written
    results_dir = (Path("results") / f"job_{job_id}").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    _write_status(
        results_dir,
        status="PENDING",
        current_step=1,
        error=None,
    )

    print(
        "[job=%s] Created workflow generation job | tasks=%s pipeline=%s rounds=%d run_after=%s calculate_after=%s",
        job_id,
        task_ids,
        pipeline_path,
        rounds,
        run_after,
        calculate_after,
    )

    asyncio.create_task(
        _run_generation_job(
            job_id=job_id,
            db=db,
            task_ids=task_ids,
            pipeline_path=pipeline_path,
            rounds=rounds,
            run_after=run_after,
            calculate_after=calculate_after,
            results_dir=results_dir,
        )
    )

    return job_id


def get_workflow_generation_status(job_id: str) -> dict:
    results_dir = (Path("results") / f"job_{job_id}").resolve()
    status_file = results_dir / "status.json"

    if not status_file.exists():
        raise KeyError("Job not found")

    return json.loads(status_file.read_text())


# -----------------------------------------------------------------------------
# Internal worker
# -----------------------------------------------------------------------------

async def _run_generation_job(
    *,
    job_id: str,
    db: Session,
    task_ids: List[str],
    pipeline_path: str,
    rounds: int,
    run_after: bool,
    calculate_after: bool,
    results_dir: Path,
):
    temp_root: Path | None = None
    original_cwd: str | None = None

    try:
        print("[job=%s] Job started", job_id)

        _write_status(
            results_dir,
            status="MATERIALIZING",
            current_step=1,
            error=None,
        )

        temp_root = Path(tempfile.mkdtemp(prefix=f"prism_job_{job_id}_"))

        # IMPORTANT: match CLI expectation → tasks/<collection>/*
        tasks_root = temp_root / "tasks" / "generated"
        tasks_root.mkdir(parents=True, exist_ok=True)

        print(
            "[job=%s] Created temp task root: %s",
            job_id,
            tasks_root,
        )

        for task_id in task_ids:
            print(
                "[job=%s] Materializing task %s",
                job_id,
                task_id,
            )
            _materialize_task(db, task_id, tasks_root)

        print("[job=%s] All tasks materialized", job_id)

        _write_status(
            results_dir,
            status="GENERATING",
            current_step=2,
            error=None,
        )

        print(
            "[job=%s] Results directory: %s",
            job_id,
            results_dir,
        )

        # Resolve pipeline path BEFORE chdir
        pipeline_path = os.path.abspath(pipeline_path)
        print(
            "[job=%s] Resolved pipeline path: %s",
            job_id,
            pipeline_path,
        )

        # PRISM expects to be run from a cwd where "tasks/..." exists
        original_cwd = os.getcwd()
        os.chdir(temp_root)

        print(
            "[job=%s] Changed working directory to %s",
            job_id,
            temp_root,
        )

        # Ensure save directories exist (scripts/operations.py does not mkdir)
        for task_id in task_ids:
            save_task_dir = results_dir / "generated" / f"task_{task_id}"
            save_task_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[job=%s] Ensured save directory exists: %s",
                job_id,
                save_task_dir,
            )

        print(
            "[job=%s] Calling generate_and_run(paths=['tasks/generated'])",
            job_id,
        )

        await generate_and_run(
            paths=["tasks/generated"],  # MUST start with "tasks"
            save_dir=str(results_dir),
            pipeline_path=pipeline_path,
            rounds=rounds,
            run_after=run_after,
            calculate_after=calculate_after,
        )

        _write_status(
            results_dir,
            status="DONE",
            current_step=3,
            error=None,
        )

        print("[job=%s] Status → DONE", job_id)

    except Exception as e:
        _write_status(
            results_dir,
            status="FAILED",
            current_step=None,
            error=str(e),
        )

        logger.exception(
            "[job=%s] Job FAILED with exception",
            job_id,
        )

    finally:
        if original_cwd is not None:
            os.chdir(original_cwd)
            print(
                "[job=%s] Restored working directory to %s",
                job_id,
                original_cwd,
            )

        if temp_root and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            print(
                "[job=%s] Cleaned up temp directory",
                job_id,
            )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _write_status(
    results_dir: Path,
    *,
    status: str,
    current_step: int | None,
    error: str | None,
):
    payload = {
        "status": status,
        "current_step": current_step,
        "error": error,
    }
    status_file = results_dir / "status.json"
    status_file.write_text(json.dumps(payload, indent=2))


def _materialize_task(db: Session, task_id: str, tasks_root: Path):
    print("Materializing task_id=%s", task_id)

    storage = get_storage()

    task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
    if not task:
        raise ValueError(f"Task not found: {task_id}")

    task_dir = tasks_root / f"task_{task.id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # task_description.txt
    (task_dir / "task_description.txt").write_text(task.description or "")

    # metadata.json
    with open(task_dir / "metadata.json", "w") as f:
        json.dump({"metric": task.metric}, f)

    queries = (
        db.query(QueryModel)
        .filter(QueryModel.task_id == task.id)
        .order_by(QueryModel.index)
        .all()
    )

    files = (
        db.query(QueryFileModel)
        .join(QueryModel, QueryFileModel.query_id == QueryModel.id)
        .filter(QueryModel.task_id == task.id)
        .all()
    )

    files_by_query: dict[int, list[QueryFileModel]] = {}
    for f in files:
        files_by_query.setdefault(f.query_id, []).append(f)

    for split in ["test", "validation"]:
        split_dir = task_dir / split
        inputs_dir = split_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        labels = ["id,label"]

        for q in queries:
            if q.split != split:
                continue

            labels.append(f"{q.index},{q.label}")

            query_input_dir = inputs_dir / str(q.index)
            query_input_dir.mkdir(parents=True, exist_ok=True)

            for f in files_by_query.get(q.id, []):
                obj = storage.get_object_stream(f.object_key)
                target = query_input_dir / f.filename
                with open(target, "wb") as out:
                    shutil.copyfileobj(obj, out)

        (split_dir / "labels.csv").write_text("\n".join(labels))
