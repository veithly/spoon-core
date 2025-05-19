import asyncio
import os

from spoon_ai.tools.storage.base_storge_tool import S3Tool


class AiozStorageTool(S3Tool):
    """S3Tool implementation for AIOZ Storage."""
    endpoint_env_key: str = "AIOZ_ENDPOINT_URL"
    aws_access_key_id: str = "AWS_ACCESS_KEY"
    aws_secret_access_key: str = "AWS_SECRET_KEY"


class AiozListBucketsTool(AiozStorageTool):
    name: str = "list_aioz_buckets"
    description: str = "List all buckets in AIOZ Storage."
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self) -> str:
        return self._list_buckets()


class UploadFileToAiozTool(AiozStorageTool):
    name: str = "upload_file_to_aioz"
    description: str = "Upload a local file to an AIOZ Storage bucket."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Target bucket name."},
            "file_path": {"type": "string", "description": "Local file path to upload."},
        },
        "required": ["bucket_name", "file_path"]
    }

    async def execute(self, bucket_name: str, file_path: str) -> str:
        return self._upload_file(bucket_name, file_path)


class DownloadFileFromAiozTool(AiozStorageTool):
    name: str = "download_file_from_aioz"
    description: str = "Download a file from AIOZ Storage."
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
        return self._download_file(bucket_name, object_key, download_path)


class DeleteAiozObjectTool(AiozStorageTool):
    name: str = "delete_aioz_object"
    description: str = "Delete an object from an AIOZ bucket."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string"},
            "object_key": {"type": "string"}
        },
        "required": ["bucket_name", "object_key"]
    }

    async def execute(self, bucket_name: str, object_key: str) -> str:
        return self._delete_object(bucket_name, object_key)


class GenerateAiozPresignedUrlTool(AiozStorageTool):
    name: str = "generate_aioz_presigned_url"
    description: str = "Generate a temporary URL to access an object in AIOZ Storage."
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
        return self._generate_presigned_url(bucket_name, object_key, expires_in)



"""
    test case
"""

##
async def test_list_buckets():
    tool = AiozListBucketsTool()
    result = await tool.execute()
    print("🧪 List Buckets Result:\n", result)


async def test_upload_file():
    bucket_name = os.getenv("BUCKET_NAME")
    file_path = "/Users/weixiaole/Downloads/file1.txt"
    with open(file_path, 'w') as f:
        f.write("This is a test file.")

    tool = UploadFileToAiozTool()
    result = await tool.execute(bucket_name=bucket_name, file_path=file_path)
    print("🧪 Upload File Result:\n", result)


async def test_download_file():
    bucket_name = os.getenv("BUCKET_NAME")
    object_key = "file1.txt"
    download_path = "/Users/weixiaole/Downloads/filex.txt"

    tool = DownloadFileFromAiozTool()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key, download_path=download_path)
    print("🧪 Download File Result:\n", result)


async def test_delete_object():
    bucket_name = os.getenv("BUCKET_NAME")
    object_key = "file1.txt"

    tool = DeleteAiozObjectTool()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key)
    print("🧪 Delete Object Result:\n", result)


async def test_generate_presigned_url():
    bucket_name = os.getenv("BUCKET_NAME")
    object_key = "file1.txt"

    tool = GenerateAiozPresignedUrlTool()
    result = await tool.execute(bucket_name=bucket_name, object_key=object_key, expires_in=600)
    print("🧪 Generate Presigned URL Result:\n", result)


if __name__ == '__main__':
    async def run_all_tests():
        await test_list_buckets()
        await test_upload_file()
        await test_generate_presigned_url()
        await test_download_file()
        await test_delete_object()

    asyncio.run(run_all_tests())
