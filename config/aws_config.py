"""
AWS Configuration for Hallmark OCR System.

Environment variables required:
- AWS_ACCESS_KEY_ID: AWS access key
- AWS_SECRET_ACCESS_KEY: AWS secret key
- AWS_REGION: AWS region (default: ap-south-1)
- S3_BUCKET_NAME: S3 bucket for image storage
- DATABASE_URL: SQLite or PostgreSQL connection string
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AWSConfig:
    """AWS configuration settings."""
    access_key_id: str
    secret_access_key: str
    region: str
    s3_bucket: str
    s3_prefix: str = "hallmark-images"

    @classmethod
    def from_env(cls) -> "AWSConfig":
        """Load configuration from environment variables."""
        return cls(
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            region=os.getenv("AWS_REGION", "ap-south-1"),
            s3_bucket=os.getenv("S3_BUCKET_NAME", "hallmark-qc-images"),
            s3_prefix=os.getenv("S3_PREFIX", "hallmark-images"),
        )

    def is_configured(self) -> bool:
        """Check if AWS credentials are configured."""
        return bool(self.access_key_id and self.secret_access_key and self.s3_bucket)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    echo: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///./hallmark_qc.db"),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )


@dataclass
class AppConfig:
    """Application configuration."""
    aws: AWSConfig
    database: DatabaseConfig
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # OCR Settings
    ocr_confidence_auto_approve: float = 0.85
    ocr_confidence_auto_reject: float = 0.50

    # Processing settings
    max_batch_size: int = 100
    image_max_size_mb: int = 10

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load full configuration from environment."""
        return cls(
            aws=AWSConfig.from_env(),
            database=DatabaseConfig.from_env(),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            ocr_confidence_auto_approve=float(os.getenv("OCR_AUTO_APPROVE", "0.85")),
            ocr_confidence_auto_reject=float(os.getenv("OCR_AUTO_REJECT", "0.50")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "100")),
            image_max_size_mb=int(os.getenv("IMAGE_MAX_SIZE_MB", "10")),
        )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
