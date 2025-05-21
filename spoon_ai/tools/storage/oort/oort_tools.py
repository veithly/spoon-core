import asyncio

from spoon_ai.tools.storage.base_storge_tool import S3Tool


class OortStorageTool(S3Tool):
    """S3Tool implementation for OORT Storage."""
    endpoint_env_key: str = "OORT_ENDPOINT_URL"
    aws_access_key_id: str = "OORT_ACCESS_KEY"
    aws_secret_access_key: str = "OORT_SECRET_KEY"


# ----------- 各功能 Function Tool 类 --------------

class OortCreateBucketTool(OortStorageTool):
    name: str = "create_oort_bucket"
    description: str = "Create a new bucket in OORT Storage."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Name of the bucket to create."},
        },
        "required": ["bucket_name"],
    }

    async def execute(self, bucket_name: str) -> str:
        return self._create_bucket(bucket_name)


class OortListBucketsTool(OortStorageTool):
    name: str = "list_oort_buckets"
    description: str = "List all buckets in OORT Storage."
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def execute(self) -> str:
        return self._list_buckets()


class OortDeleteBucketTool(OortStorageTool):
    name: str = "delete_oort_bucket"
    description: str = "Delete a bucket from OORT Storage."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Name of the bucket to delete."},
        },
        "required": ["bucket_name"],
    }

    async def execute(self, bucket_name: str) -> str:
        return self._delete_bucket(bucket_name)


class OortListObjectsTool(OortStorageTool):
    name: str = "list_oort_objects"
    description: str = "List objects in a bucket (up to 1000)."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Bucket name."},
            "prefix": {"type": "string", "description": "Prefix filter for objects.", "default": ""},
        },
        "required": ["bucket_name"],
    }

    async def execute(self, bucket_name: str, prefix: str = "") -> str:
        return "\n".join(self._list_objects(bucket_name, prefix))


class OortUploadFileTool(OortStorageTool):
    name: str = "upload_file_to_oort"
    description: str = "Upload a local file to an OORT Storage bucket."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Target bucket name."},
            "file_path": {"type": "string", "description": "Local file path to upload."},
        },
        "required": ["bucket_name", "file_path"],
    }

    async def execute(self, bucket_name: str, file_path: str) -> str:
        return self._upload_file(bucket_name, file_path)


class OortDownloadFileTool(OortStorageTool):
    name: str = "download_file_from_oort"
    description: str = "Download an object from OORT Storage to local path."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Bucket name."},
            "object_key": {"type": "string", "description": "Key of the object to download."},
            "download_path": {"type": "string", "description": "Local path to save the downloaded file."},
        },
        "required": ["bucket_name", "object_key", "download_path"],
    }

    async def execute(self, bucket_name: str, object_key: str, download_path: str) -> str:
        return self._download_file(bucket_name, object_key, download_path)


class OortDeleteObjectTool(OortStorageTool):
    name: str = "delete_oort_object"
    description: str = "Delete a single object from a bucket."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Bucket name."},
            "object_key": {"type": "string", "description": "Key of the object to delete."},
        },
        "required": ["bucket_name", "object_key"],
    }

    async def execute(self, bucket_name: str, object_key: str) -> str:
        return self._delete_object(bucket_name, object_key)


class OortDeleteObjectsTool(OortStorageTool):
    name: str = "delete_oort_objects"
    description: str = "Delete multiple objects from a bucket."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Bucket name."},
            "object_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of object keys to delete.",
            },
        },
        "required": ["bucket_name", "object_keys"],
    }

    async def execute(self, bucket_name: str, object_keys: list) -> str:
        return self._delete_objects(bucket_name, object_keys)


class OortGeneratePresignedUrlTool(OortStorageTool):
    name: str = "generate_oort_presigned_url"
    description: str = "Generate a presigned URL for an object in OORT Storage."
    parameters: dict = {
        "type": "object",
        "properties": {
            "bucket_name": {"type": "string", "description": "Bucket name."},
            "object_key": {"type": "string", "description": "Object key."},
            "expires_in": {"type": "integer", "description": "Expiry time in seconds.", "default": 3600},
        },
        "required": ["bucket_name", "object_key"],
    }

    async def execute(self, bucket_name: str, object_key: str, expires_in: int = 3600) -> str:
        return self._generate_presigned_url(bucket_name, object_key, expires_in)


async def test_create_bucket():
    tool = OortCreateBucketTool()
    result = await tool.execute("mytestbucket")
    print("🧪 Create Bucket Result:\n", result)


async def test_list_buckets():
    tool = OortListBucketsTool()
    result = await tool.execute()
    print("🧪 List Buckets Result:\n", result)


async def test_delete_bucket():
    tool = OortDeleteBucketTool()
    result = await tool.execute("mytestbucket")
    print("🧪 Delete Bucket Result:\n", result)


async def test_list_objects():
    tool = OortListObjectsTool()
    result = await tool.execute("mytestbucket", prefix="")
    print("🧪 List Objects Result:\n", result)


async def test_upload_file():
    tool = OortUploadFileTool()
    result = await tool.execute("mytestbucket", "/Users/weixiaole/Downloads/file1.txt")
    print("🧪 Upload File Result:\n", result)


async def test_download_file():
    tool = OortDownloadFileTool()
    result = await tool.execute("mytestbucket", "file1.txt", "/Users/weixiaole/Downloads/file111.txt")
    print("🧪 Download File Result:\n", result)


async def test_delete_object():
    tool = OortDeleteObjectTool()
    result = await tool.execute("mytestbucket", "file1.txt")
    print("🧪 Delete Object Result:\n", result)


async def test_delete_objects():
    tool = OortDeleteObjectsTool()
    keys = ["file1.txt", "file2.txt"]
    result = await tool.execute("mytestbucket", keys)
    print("🧪 Delete Objects Result:\n", result)


async def test_generate_presigned_url():
    tool = OortGeneratePresignedUrlTool()
    result = await tool.execute("mytestbucket", "file1.txt", expires_in=600)
    print("🧪 Generate Presigned URL Result:\n", result)


# 统一运行所有测试
async def run_all_tests():
    await test_create_bucket()
    # await test_list_buckets()
    await test_upload_file()
    # await test_list_objects()
    # await test_generate_presigned_url()
    await test_download_file()
    # await test_delete_object()
    # await test_delete_objects()
    # await test_delete_bucket()


if __name__ == '__main__':
    asyncio.run(run_all_tests())
