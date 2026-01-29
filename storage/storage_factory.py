from storage.minio_storage import MinIOStorage
from utils.constants import SETTINGS


def get_storage():
    return MinIOStorage(
        endpoint=SETTINGS.MINIO_ENDPOINT,
        access_key=SETTINGS.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO_SECRET_KEY,
        bucket=SETTINGS.MINIO_BUCKET,
        secure=SETTINGS.MINIO_SECURE,
    )
