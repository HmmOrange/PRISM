from datetime import datetime, timedelta
from minio import Minio
from minio.datatypes import PostPolicy

from storage.base_storage import BaseStorage


class MinIOStorage(BaseStorage):
    def __init__(
        self,
        endpoint_external: str,  # localhost:9000
        endpoint_internal: str,   # minio:9000
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket

        # Internal-only client
        self.client = Minio(
            endpoint_internal,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    # ---------- UPLOAD (POST) ----------
    def generate_presigned_upload_post(
        self,
        object_key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> dict:
        policy = PostPolicy(
            self.bucket,
            datetime.utcnow() + timedelta(seconds=expires_in),
        )

        policy.add_equals_condition("key", object_key)
        policy.add_starts_with_condition("Content-Type", content_type)

        fields = self.client.presigned_post_policy(policy)

        return {
            "url": f"/storage/upload/{self.bucket}",
            "fields": fields,
        }

    # ---------- DOWNLOAD (STREAM) ----------
    def get_object_stream(self, object_key: str):
        return self.client.get_object(
            self.bucket,
            object_key,
        )
