from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from minio.error import S3Error

from storage.storage_factory import get_storage

router = APIRouter(prefix="/storage", tags=["Storage"])


@router.get("/download")
def download_file(object_key: str):
    storage = get_storage()

    try:
        stat = storage.client.stat_object(
            storage.bucket,
            object_key,
        )

        obj = storage.client.get_object(
            storage.bucket,
            object_key,
        )

        return StreamingResponse(
            obj,
            media_type=stat.content_type or "application/octet-stream",
            headers={
                "Content-Length": str(stat.size),
                "Content-Disposition": f'inline; filename="{object_key.split("/")[-1]}"',
                "Cache-Control": "no-cache",
            },
        )

    except S3Error:
        raise HTTPException(status_code=404, detail="File not found")
