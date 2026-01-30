from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseStorage(ABC):
    # ---------- UPLOAD ----------
    @abstractmethod
    def generate_presigned_upload_post(
        self,
        object_key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> Dict[str, Any]:
        """
        Generate a presigned POST policy for uploading an object.

        Returns:
        {
          "url": "/storage/upload/<bucket>",
          "fields": {
            ...
          }
        }
        """
        raise NotImplementedError

    # ---------- DOWNLOAD (PROXY ONLY) ----------
    @abstractmethod
    def get_object_stream(self, object_key: str):
        """
        Return a readable stream for an object.
        Used ONLY by backend download proxy.
        """
        raise NotImplementedError
