import asyncio
import os
import traceback

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from spoon_ai.tools.base import BaseTool

"""
    Currently supported:
    UploadFileTool, ListBucketsTool, DownloadFileTool, 
    DeleteObjectTool, GeneratePresignedUrlTool
"""


class AiozStorageTool(BaseTool):
    """Base tool for interacting with AIOZ Storage."""

    def _get_s3_client(self):
        """Returns a boto3 S3 client initialized for AIOZ Storage."""
        aioz_access_key = os.getenv("AIOZ_ACCESS_KEY")
        aioz_secret_key = os.getenv("AIOZ_SECRET_KEY")
        if not aioz_access_key or not aioz_secret_key:
            raise ValueError("Missing AIOZ credentials in environment variables!")
        return boto3.client(
            's3',
            # region_name='us-east-1',
            aws_access_key_id=aioz_access_key,
            aws_secret_access_key=aioz_secret_key,
            endpoint_url="https://s3.aiozstorage.network",
            config=Config(s3={'addressing_style': 'path'}),
            # verify=False
        )


class UploadFileTool(AiozStorageTool):
    name: str = "upload_file_to_aioz"
    description: str = "Upload a local file to AIOZ Storage bucket"
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Name of the AIOZ storage bucket"},
            "file_path": {"type": "string", "description": "Local path to the file to upload"}
        },
        "required": ["bucket_name", "file_path"]
    }

    async def execute(self, bucket_name: str, file_path: str) -> str:
        s3 = self._get_s3_client()
        try:
            bucket = s3.Bucket(bucket_name)
            object_key = os.path.basename(file_path)
            with open(file_path, 'rb') as data:
                result = bucket.put_object(Key=object_key, Body=data)
            return f"✅ File '{object_key}' uploaded to bucket '{bucket_name}' successfully."
        except ClientError as e:
            return f"❌ Upload failed: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"


class ListBucketsTool(AiozStorageTool):
    name: str = "list_aioz_buckets"
    description: str = "List all buckets in AIOZ Storage"
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self) -> str:
        s3 = self._get_s3_client()
        try:
            buckets = s3.list_buckets()
            if not buckets["Buckets"]:
                return "📦 No buckets found."
            return "\n".join([f"📁 {b['Name']}" for b in buckets["Buckets"]])
        except ClientError as e:
            return f"❌ Error listing buckets: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"


class DownloadFileTool(AiozStorageTool):
    name: str = "download_file_from_aioz"
    description: str = "Download a file from AIOZ Storage bucket"
    parameters: dict = {
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
            return f"✅ File '{object_key}' downloaded to '{download_path}'."
        except ClientError as e:
            return f"❌ Download failed: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"


class DeleteObjectTool(AiozStorageTool):
    name: str = "delete_aioz_object"
    description: str = "Delete an object from an AIOZ bucket"
    parameters: dict = {
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
        except ClientError as e:
            return f"❌ Deletion failed: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"


class GeneratePresignedUrlTool(AiozStorageTool):
    name: str = "generate_aioz_presigned_url"
    description: str = "Generate a temporary URL to access an object in AIOZ"
    parameters: dict = {
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
        except ClientError as e:
            return f"❌ Failed to generate URL: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"



if __name__ == '__main__':
    async def create_aioz_bucket():
        print(os.getenv("AIOZ_ACCESS_KEY"))
        bucket_name = "neo-test-bucket2"

        tool = ListBucketsTool()
        result = await tool.execute()
        print(result)

    asyncio.run(create_aioz_bucket())