"""
MinIO 对象存储服务
提供文件上传、下载、删除等操作的统一接口
"""

from typing import Dict, Iterator, List, Optional
from minio import Minio
from minio.error import S3Error

from config import settings


class MinIOService:
    """MinIO 对象存储服务"""

    # 私有类变量，缓存 MinIO 客户端实例
    _client: Optional[Minio] = None

    @classmethod
    def _get_client(cls) -> Minio:
        """
        获取 MinIO 客户端实例（单例模式）

        Returns:
            Minio: MinIO 客户端实例
        """
        if cls._client is None:
            cls._client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
        return cls._client

    @staticmethod
    def ensure_bucket_exists(bucket_name: Optional[str] = None) -> None:
        """
        确保存储桶存在，不存在则创建

        Args:
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

    @staticmethod
    def upload_file(
        file_path: str,
        object_name: str,
        bucket_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        上传本地文件到 MinIO

        Args:
            file_path: 本地文件路径
            object_name: MinIO 中的对象名称
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME
            metadata: 文件元数据

        Returns:
            str: 上传的对象名称

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        MinIOService.ensure_bucket_exists(bucket)

        client.fput_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=file_path,
            metadata=metadata
        )

        return object_name

    @staticmethod
    def upload_from_bytes(
        data: bytes,
        object_name: str,
        bucket_name: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
       上传字节流到 MinIO

        Args:
            data: 文件字节数据
            object_name: MinIO 中的对象名称
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME
            content_type: 内容类型
            metadata: 文件元数据

        Returns:
            str: 上传的对象名称

        Raises:
            S3Error: MinIO 操作错误
        """
        from io import BytesIO

        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        MinIOService.ensure_bucket_exists(bucket)

        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
            metadata=metadata
        )

        return object_name

    @staticmethod
    def download_file(
        object_name: str,
        file_path: str,
        bucket_name: Optional[str] = None
    ) -> None:
        """
        从 MinIO 下载文件到本地路径

        Args:
            object_name: MinIO 中的对象名称
            file_path: 本地保存路径
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        client.fget_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=file_path
        )

    @staticmethod
    def get_file_bytes(
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> bytes:
        """
        从 MinIO 获取文件字节流

        Args:
            object_name: MinIO 中的对象名称
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Returns:
            bytes: 文件字节数据

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        response = client.get_object(
            bucket_name=bucket,
            object_name=object_name
        )

        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def iter_object_chunks(
        object_name: str,
        chunk_size: int = 32 * 1024,
        bucket_name: Optional[str] = None,
    ) -> Iterator[bytes]:
        """
        按块迭代 MinIO 对象内容，用于 StreamingResponse，避免大文件整包读入内存。

        Args:
            object_name: MinIO 中的对象名称
            chunk_size: 每块字节数
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Yields:
            bytes: 文件分块数据
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        response = client.get_object(
            bucket_name=bucket,
            object_name=object_name,
        )
        try:
            for chunk in response.stream(chunk_size):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def delete_file(
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        删除 MinIO 中的文件

        Args:
            object_name: MinIO 中的对象名称
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Returns:
            bool: 删除成功返回 True

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        client.remove_object(
            bucket_name=bucket,
            object_name=object_name
        )

        return True

    @staticmethod
    def file_exists(
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        检查文件是否存在

        Args:
            object_name: MinIO 中的对象名称
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Returns:
            bool: 文件存在返回 True

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        try:
            client.stat_object(
                bucket_name=bucket,
                object_name=object_name
            )
            return True
        except S3Error:
            return False

    @staticmethod
    def get_presigned_url(
        object_name: str,
        expires: int = 3600,
        bucket_name: Optional[str] = None
    ) -> str:
        """
        获取预签名 URL，用于直接访问文件

        Args:
            object_name: MinIO 中的对象名称
            expires: URL 过期时间（秒），默认 1 小时
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Returns:
            str: 预签名 URL

        Raises:
            S3Error: MinIO 操作错误
        """
        from datetime import timedelta

        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        return client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(seconds=expires)
        )

    @staticmethod
    def list_files(
        prefix: str = "",
        bucket_name: Optional[str] = None
    ) -> List[str]:
        """
        列出指定前缀的文件

        Args:
            prefix: 对象名称前缀，用于过滤
            bucket_name: 存储桶名称，默认使用配置中的 MINIO_BUCKET_NAME

        Returns:
            List[str]: 对象名称列表

        Raises:
            S3Error: MinIO 操作错误
        """
        client = MinIOService._get_client()
        bucket = bucket_name or settings.MINIO_BUCKET_NAME

        objects = client.list_objects(
            bucket_name=bucket,
            prefix=prefix,
            recursive=True
        )

        return [obj.object_name for obj in objects]
