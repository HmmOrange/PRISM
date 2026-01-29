from datetime import timedelta
from minio import Minio

from storage.base_storage import BaseStorage


class MinIOStorage(BaseStorage):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket

        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def generate_presigned_upload_url(
        self,
        object_key: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned PUT URL for uploading a single object.

        IMPORTANT:
        - MinIO does NOT support `content_type` in presigned_put_object
        - Client must send Content-Type during PUT upload
        """
        return self.client.presigned_put_object(
            bucket_name=self.bucket,
            object_name=object_key,
            expires=timedelta(seconds=expires_in),
        )
