import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional
from app.config import settings


class StorageAdapter(ABC):
    @abstractmethod
    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        pass

    @abstractmethod
    async def get_bytes(self, key: str) -> bytes:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def get_presigned_url(self, key: str, expires_sec: int = 3600) -> str:
        pass


class LocalDiskStorageAdapter(StorageAdapter):
    """
    Zero-cost local filesystem storage adapter for development and demo mode.
    Stores files under settings.LOCAL_STORAGE_DIR.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.LOCAL_STORAGE_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_path(self, key: str) -> str:
        safe_key = key.replace("\\", "/").strip("/").replace("..", "_")
        full_path = os.path.join(self.base_dir, safe_key)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._get_path(key)
        with open(path, "wb") as f:
            f.write(data)
        return key

    async def get_bytes(self, key: str) -> bytes:
        path = self._get_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Key {key} not found in local storage.")
        with open(path, "rb") as f:
            return f.read()

    async def delete(self, key: str) -> bool:
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return False

    async def get_presigned_url(self, key: str, expires_sec: int = 3600) -> str:
        return f"{settings.PUBLIC_API_URL}/api/v1/media/{key}"


class CloudflareR2StorageAdapter(StorageAdapter):
    """
    Production S3-compatible adapter targeting Cloudflare R2 ($0 egress fees).
    Falls back gracefully to LocalDiskStorageAdapter if credentials are not configured.
    """

    def __init__(self):
        self.endpoint_url = settings.R2_ENDPOINT_URL
        self.access_key = settings.R2_ACCESS_KEY_ID
        self.secret_key = settings.R2_SECRET_ACCESS_KEY
        self.bucket = settings.R2_BUCKET_NAME
        self.fallback = LocalDiskStorageAdapter()

        self.configured = bool(self.endpoint_url and self.access_key and self.secret_key)

    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        if not self.configured:
            return await self.fallback.put_bytes(key, data, content_type)
        # S3 / R2 async put via boto3 in thread
        import asyncio
        import boto3

        def _sync_put():
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
            )
            client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
            return key

        return await asyncio.to_thread(_sync_put)

    async def get_bytes(self, key: str) -> bytes:
        if not self.configured:
            return await self.fallback.get_bytes(key)
        import asyncio
        import boto3

        def _sync_get():
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
            )
            res = client.get_object(Bucket=self.bucket, Key=key)
            return res["Body"].read()

        return await asyncio.to_thread(_sync_get)

    async def delete(self, key: str) -> bool:
        if not self.configured:
            return await self.fallback.delete(key)
        import asyncio
        import boto3

        def _sync_delete():
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
            )
            client.delete_object(Bucket=self.bucket, Key=key)
            return True

        return await asyncio.to_thread(_sync_delete)

    async def get_presigned_url(self, key: str, expires_sec: int = 3600) -> str:
        if not self.configured:
            return await self.fallback.get_presigned_url(key, expires_sec)
        import asyncio
        import boto3

        def _sync_url():
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
            )
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_sec,
            )

        return await asyncio.to_thread(_sync_url)


def get_storage_adapter() -> StorageAdapter:
    if settings.STORAGE_PROVIDER == "r2":
        return CloudflareR2StorageAdapter()
    return LocalDiskStorageAdapter()
