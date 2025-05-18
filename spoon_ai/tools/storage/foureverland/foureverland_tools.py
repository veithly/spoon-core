import asyncio
import os
import boto3
from botocore.config import Config
from spoon_ai.tools.base import BaseTool


class FourEverlandStorageTool(BaseTool):
    """Base tool for interacting with 4EVERLAND Storage."""

    def _get_s3_client(self):
        access_key = os.getenv("FOREVERLAND_ACCESS_KEY")
        secret_key = os.getenv("FOREVERLAND_SECRET_KEY")
        endpoint_url = os.getenv("FOREVERLAND_ENDPOINT_URL", "https://endpoint.4everland.co")
        if not access_key or not secret_key:
            raise ValueError("Missing 4EVERLAND credentials in environment variables!")

        return boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name='us-east-1',
            config=Config(s3={'addressing_style': 'path'})
        )

    def _get_s3_resource(self):
        access_key = os.getenv("FOREVERLAND_ACCESS_KEY")
        secret_key = os.getenv("FOREVERLAND_SECRET_KEY")
        endpoint_url = os.getenv("FOREVERLAND_ENDPOINT_URL", "https://endpoint.4everland.co")

        return boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name='us-east-1',
            config=Config(s3={'addressing_style': 'path'})
        )


class UploadFileToFourEverland(FourEverlandStorageTool):
    name: str = "upload_file_to_4everland"
    description: str = "Upload a file to 4EVERLAND Storage"
    parameters: str = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string"},
            "file_path": {"type": "string"}
        },
        "required": ["bucket_name", "file_path"]
    }

    async def execute(self, bucket_name: str, file_path: str) -> str:
        s3 = self._get_s3_resource()
        try:
            object_key = os.path.basename(file_path)
            bucket = s3.Bucket(bucket_name)
            obj = bucket.Object(object_key)

            with open(file_path, 'rb') as f:
                obj.put(Body=f)
                obj.wait_until_exists()
            return f"✅ File '{object_key}' uploaded to '{bucket_name}'"
        except Exception as e:
            return f"❌ Upload failed: {e}"


class ListFourEverlandBuckets(FourEverlandStorageTool):
    name: str = "list_4everland_buckets"
    description: str = "List all buckets in 4EVERLAND Storage"
    parameters: str = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self) -> str:
        s3 = self._get_s3_client()
        try:
            buckets = s3.list_buckets()
            if not buckets.get("Buckets"):
                return "📦 No buckets found."
            return "\n".join([f"📁 {b['Name']}" for b in buckets["Buckets"]])
        except Exception as e:
            return f"❌ Failed to list buckets: {e}"


class DownloadFileFromFourEverland(FourEverlandStorageTool):
    name: str = "download_file_from_4everland"
    description: str = "Download a file from 4EVERLAND Storage"
    parameters: str = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string"},
            "object_key": {"type": "string"},
            "download_path": {"type": "string"}
        },
        "required": ["bucket_name", "object_key", "download_path"]
    }

    async def execute(self, bucket_name: str, object_key: str, download_path: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.download_file(bucket_name, object_key, download_path)
            return f"✅ Downloaded '{object_key}' to '{download_path}'"
        except Exception as e:
            return f"❌ Download failed: {e}"


class DeleteFourEverlandObject(FourEverlandStorageTool):
    name: str = "delete_4everland_object"
    description: str = "Delete an object from 4EVERLAND Storage"
    parameters: str = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string"},
            "object_key": {"type": "string"}
        },
        "required": ["bucket_name", "object_key"]
    }

    async def execute(self, bucket_name: str, object_key: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.delete_object(Bucket=bucket_name, Key=object_key)
            return f"🗑️ Object '{object_key}' deleted from bucket '{bucket_name}'."
        except Exception as e:
            return f"❌ Deletion failed: {e}"


class GenerateFourEverlandPresignedUrl(FourEverlandStorageTool):
    name: str = "generate_4everland_presigned_url"
    description: str = "Generate a temporary URL to access a 4EVERLAND object"
    parameters: str = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string"},
            "object_key": {"type": "string"},
            "expires_in": {"type": "integer", "default": 3600}
        },
        "required": ["bucket_name", "object_key"]
    }

    async def execute(self, bucket_name: str, object_key: str, expires_in: int = 3600) -> str:
        s3 = self._get_s3_client()
        try:
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expires_in
            )
            return f"🔗 Temporary URL:\n{url}"
        except Exception as e:
            return f"❌ Failed to generate URL: {e}"


async def test_list_foureverland_buckets():
    tool = ListFourEverlandBuckets()
    result = await tool.execute()
    print("🧪 List 4EVERLAND Buckets:\n", result)


async def test_upload_file_to_foureverland():
    bucket_name = os.getenv("FOREVERLAND_BUCKET_NAME")
    file_path = "/Users/weixiaole/Downloads/file1.txt"

    # 创建测试文件
    with open(file_path, 'w') as f:
        f.write("🌍 4EVERLAND test content")

    tool = UploadFileToFourEverland()
    result = await tool.execute(bucket_name=bucket_name, file_path=file_path)
    print("🧪 Upload File Result:\n", result)


async def test_generate_presigned_url_foureverland():
    bucket_name = os.getenv("FOREVERLAND_BUCKET_NAME")
    object_key = "file1.txt"

    tool = GenerateFourEverlandPresignedUrl()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key, expires_in=600)
    print("🧪 Generate Presigned URL Result:\n", result)


async def test_download_file_from_foureverland():
    bucket_name = os.getenv("FOREVERLAND_BUCKET_NAME")
    object_key = "file1.txt"
    download_path = "/Users/weixiaole/Downloads/test_file_downloaded.txt"

    tool = DownloadFileFromFourEverland()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key, download_path=download_path)
    print("🧪 Download File Result:\n", result)


async def test_delete_foureverland_object():
    bucket_name = os.getenv("FOREVERLAND_BUCKET_NAME")
    object_key = "file1.txt"

    tool = DeleteFourEverlandObject()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key)
    print("🧪 Delete Object Result:\n", result)


if __name__ == '__main__':
    async def run_all_foureverland_tests():
        await test_list_foureverland_buckets()
        await test_upload_file_to_foureverland()
        await test_generate_presigned_url_foureverland()
        await test_download_file_from_foureverland()
        await test_delete_foureverland_object()


    asyncio.run(run_all_foureverland_tests())
