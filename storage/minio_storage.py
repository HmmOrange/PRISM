from datetime import timedelta
from minio import Minio

from storage.base_storage import BaseStorage


class MinIOStorage(BaseStorage):
    def __init__(
        self,
        endpoint_internal: str,
        endpoint_external: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.endpoint_internal = endpoint_internal
        self.endpoint_external = endpoint_external

        self.client = Minio(
            endpoint_internal,
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

        Notes:
        - MinIO presigned_put_object does NOT accept content_type
        - Client must send Content-Type during PUT
        """
        url = self.client.presigned_put_object(
            self.bucket,
            object_key,
            expires=timedelta(seconds=expires_in)
        )

        # Rewrite Docker-internal hostname → browser-reachable hostname
        return url.replace(
            self.endpoint_internal,
            self.endpoint_external,
        )
