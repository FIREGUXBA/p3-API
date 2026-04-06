"""
应用配置：从环境变量与可选的 .env 读取，不依赖 pydantic-settings。
"""

import os
from pathlib import Path


def _load_dotenv_simple() -> None:
    """若存在项目根目录 .env，则载入到 os.environ（不覆盖已有变量）。"""
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


_load_dotenv_simple()


class Settings:
    """应用配置（与原先 pydantic BaseSettings 字段一致）。"""

    MINIO_ENDPOINT: str = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.environ.get("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY: str = os.environ.get("MINIO_SECRET_KEY", "password123")
    MINIO_SECURE: bool = _env_bool("MINIO_SECURE", False)
    MINIO_BUCKET_NAME: str = os.environ.get("MINIO_BUCKET_NAME", "panorama-3dgs")


settings = Settings()
