from abc import ABC, abstractmethod


class BaseStorage(ABC):
    @abstractmethod
    def generate_presigned_upload_url(
        self,
        object_key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> str:
        pass
