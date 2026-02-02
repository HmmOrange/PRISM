from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.session import get_db
from db.services.workflow.workflow_generation_service import (
    start_workflow_generation_job,
    get_workflow_generation_status,
)

router = APIRouter(prefix="/workflows", tags=["Workflows"])


@router.post("/generate")
async def generate_workflows(
    payload: dict,
    db: Session = Depends(get_db),
):
    """
    Start a workflow generation job.

    Expected payload:
    {
        "task_ids": [str, ...],
        "pipeline_path": str,
        "rounds": int,
        "run_after": bool,
        "calculate_after": bool
    }
    """
    try:
        job_id = start_workflow_generation_job(
            db=db,
            task_ids=payload["task_ids"],
            pipeline_path=payload["pipeline_path"],
            rounds=payload.get("rounds", 1),
            run_after=payload.get("run_after", False),
            calculate_after=payload.get("calculate_after", False),
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        # IMPORTANT: stringify safely
        raise HTTPException(status_code=500, detail=str(e))

    return {"job_id": job_id}


@router.get("/generate/{job_id}/status")
async def get_generation_status(job_id: str):
    """
    Get the status of a workflow generation job.
    """
    try:
        status = get_workflow_generation_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    return status
