"""
Storage Service for Hallmark OCR System.

Handles file storage operations:
- AWS S3 for production deployment
- Local filesystem for development
"""

import os
import io
import uuid
from datetime import datetime
from typing import Optional, Tuple, BinaryIO
from pathlib import Path
from PIL import Image

# Try to import boto3, fall back gracefully if not available
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))
from aws_config import AWSConfig, get_config


class StorageService:
    """
    Storage service that supports both S3 and local filesystem.
    Uses S3 when AWS credentials are configured, otherwise falls back to local storage.
    """

    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or get_config().aws
        self.use_s3 = BOTO3_AVAILABLE and self.config.is_configured()
        self._s3_client = None

        # Local storage directory
        self.local_storage_path = Path("./uploads")
        self.local_storage_path.mkdir(parents=True, exist_ok=True)

        if self.use_s3:
            self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client."""
        if not BOTO3_AVAILABLE:
            return

        try:
            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize S3 client: {e}")
            self.use_s3 = False

    def _generate_key(self, tag_id: str, filename: str, prefix: str = "") -> str:
        """Generate a unique key for the file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        ext = os.path.splitext(filename)[1].lower() or ".jpg"
        unique_id = uuid.uuid4().hex[:8]

        if prefix:
            return f"{self.config.s3_prefix}/{prefix}/{timestamp}/{tag_id}_{unique_id}{ext}"
        return f"{self.config.s3_prefix}/{timestamp}/{tag_id}_{unique_id}{ext}"

    def upload_image(
        self,
        file_data: bytes,
        tag_id: str,
        filename: str,
        prefix: str = "originals"
    ) -> Tuple[str, Optional[str]]:
        """
        Upload an image file.

        Args:
            file_data: Image file bytes
            tag_id: Tag ID for the item
            filename: Original filename
            prefix: Subdirectory prefix (e.g., "originals", "processed")

        Returns:
            Tuple of (storage_path, public_url)
            - storage_path: Path/key where file is stored
            - public_url: Public URL if available (S3 only)
        """
        key = self._generate_key(tag_id, filename, prefix)

        if self.use_s3:
            return self._upload_to_s3(file_data, key, filename)
        else:
            return self._upload_to_local(file_data, key)

    def _upload_to_s3(self, file_data: bytes, key: str, filename: str) -> Tuple[str, Optional[str]]:
        """Upload file to S3."""
        try:
            content_type = self._get_content_type(filename)

            self._s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=file_data,
                ContentType=content_type,
            )

            # Generate public URL (if bucket is public or using presigned URLs)
            url = f"https://{self.config.s3_bucket}.s3.{self.config.region}.amazonaws.com/{key}"

            return key, url

        except Exception as e:
            print(f"S3 upload failed: {e}, falling back to local storage")
            return self._upload_to_local(file_data, key)

    def _upload_to_local(self, file_data: bytes, key: str) -> Tuple[str, Optional[str]]:
        """Upload file to local storage."""
        # Convert S3-style key to local path
        local_path = self.local_storage_path / key.replace("/", os.sep)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(file_data)

        # Return path relative to project root and a local URL
        relative_path = str(local_path)
        local_url = f"/uploads/{key}"

        return relative_path, local_url

    def upload_pil_image(
        self,
        image: Image.Image,
        tag_id: str,
        prefix: str = "processed",
        format: str = "JPEG"
    ) -> Tuple[str, Optional[str]]:
        """
        Upload a PIL Image object.

        Args:
            image: PIL Image to upload
            tag_id: Tag ID for the item
            prefix: Subdirectory prefix
            format: Image format (JPEG, PNG)

        Returns:
            Tuple of (storage_path, public_url)
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95)
        buffer.seek(0)

        ext = ".jpg" if format.upper() == "JPEG" else f".{format.lower()}"
        filename = f"{tag_id}{ext}"

        return self.upload_image(buffer.getvalue(), tag_id, filename, prefix)

    def get_image(self, path: str) -> Optional[bytes]:
        """
        Get image data from storage.

        Args:
            path: Storage path/key

        Returns:
            Image bytes or None if not found
        """
        if self.use_s3 and not path.startswith(str(self.local_storage_path)):
            return self._get_from_s3(path)
        return self._get_from_local(path)

    def _get_from_s3(self, key: str) -> Optional[bytes]:
        """Get file from S3."""
        try:
            response = self._s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=key
            )
            return response["Body"].read()
        except Exception as e:
            print(f"S3 get failed: {e}")
            return None

    def _get_from_local(self, path: str) -> Optional[bytes]:
        """Get file from local storage."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                # Try with local storage prefix
                file_path = self.local_storage_path / path.lstrip("/").replace("uploads/", "")

            if file_path.exists():
                return file_path.read_bytes()
            return None
        except Exception as e:
            print(f"Local get failed: {e}")
            return None

    def get_presigned_url(self, path: str, expiration: int = 3600) -> Optional[str]:
        """
        Get a presigned URL for accessing a file.

        Args:
            path: Storage path/key
            expiration: URL expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL or None
        """
        if not self.use_s3:
            # For local storage, return the local URL
            return f"/uploads/{path}" if not path.startswith("/") else path

        try:
            url = self._s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.s3_bucket, "Key": path},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Failed to generate presigned URL: {e}")
            return None

    def delete_image(self, path: str) -> bool:
        """
        Delete an image from storage.

        Args:
            path: Storage path/key

        Returns:
            True if deleted successfully
        """
        if self.use_s3 and not path.startswith(str(self.local_storage_path)):
            return self._delete_from_s3(path)
        return self._delete_from_local(path)

    def _delete_from_s3(self, key: str) -> bool:
        """Delete file from S3."""
        try:
            self._s3_client.delete_object(
                Bucket=self.config.s3_bucket,
                Key=key
            )
            return True
        except Exception as e:
            print(f"S3 delete failed: {e}")
            return False

    def _delete_from_local(self, path: str) -> bool:
        """Delete file from local storage."""
        try:
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Local delete failed: {e}")
            return False

    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension."""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return content_types.get(ext, "application/octet-stream")

    @property
    def storage_type(self) -> str:
        """Return the current storage type being used."""
        return "s3" if self.use_s3 else "local"


# Global storage instance
_storage: Optional[StorageService] = None


def get_storage() -> StorageService:
    """Get the global storage service instance."""
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
