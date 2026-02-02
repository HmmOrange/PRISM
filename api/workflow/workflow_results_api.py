from fastapi import APIRouter, HTTPException
from pathlib import Path
from fastapi.responses import FileResponse
import json

router = APIRouter(prefix="/workflows", tags=["Workflows"])


@router.get("/generate/{job_id}/results")
def get_workflow_results(job_id: str):
    """
    List generated workflow files for a job.
    """
    base_dir = (Path("results") / f"job_{job_id}").resolve()

    if not base_dir.exists():
        raise HTTPException(status_code=404, detail="Job results not found")

    results = []

    for path in base_dir.rglob("workflow*.py"):
        results.append({
            "task": path.parent.name,
            "relative_path": str(path.relative_to(base_dir)),
            "filename": path.name,
            "download_url": (
                f"/workflows/generate/{job_id}/results/"
                f"{path.relative_to(base_dir)}"
            ),
        })

    return results


@router.get("/generate/{job_id}/results/{file_path:path}")
def download_workflow(job_id: str, file_path: str):
    base_dir = (Path("results") / f"job_{job_id}").resolve()
    path = base_dir / file_path

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        media_type="text/x-python",
        filename=path.name,
    )


@router.get("/generate/{job_id}/status")
def get_generation_status(job_id: str):
    base_dir = (Path("results") / f"job_{job_id}").resolve()
    status_file = base_dir / "status.json"

    if status_file.exists():
        return json.loads(status_file.read_text())

    raise HTTPException(status_code=404, detail="Job not found")
