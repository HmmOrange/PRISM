from storage.minio_storage import MinIOStorage
from utils.constants import SETTINGS


def get_storage():
    return MinIOStorage(
        endpoint_external=SETTINGS.MINIO_EXTERNAL_ENDPOINT,
        endpoint_internal=SETTINGS.MINIO_INTERNAL_ENDPOINT,
        access_key=SETTINGS.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO_SECRET_KEY,
        bucket=SETTINGS.MINIO_BUCKET,
        secure=SETTINGS.MINIO_SECURE,
    )
