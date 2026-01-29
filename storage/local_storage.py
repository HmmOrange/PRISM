import os
from .base_storage import BaseStorage

class LocalStorage(BaseStorage):
    def _path(self, uri: str) -> str:
        return uri.replace("file://", "")

    def read(self, uri: str):
        return open(self._path(uri), "rb")

    def write(self, uri: str, data: bytes):
        path = self._path(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def exists(self, uri: str) -> bool:
        return os.path.exists(self._path(uri))
