import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from spoon_ai.tools.base import BaseTool
from abc import ABC


class S3Tool(BaseTool, ABC):
    """Abstract base class for S3-compatible storage tools with rich capabilities."""

    endpoint_env_key: str = None
    aws_access_key_id: str = None
    aws_secret_access_key: str = None

    def _get_env(self, name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    def _get_s3_client(self):
        return boto3.client(
            's3',
            aws_access_key_id=self._get_env(self.aws_access_key_id),
            aws_secret_access_key=self._get_env(self.aws_secret_access_key),
            endpoint_url=self._get_env(self.endpoint_env_key),
            config=Config(s3={'addressing_style': 'path'})
        )

    def _get_s3_resource(self):
        return boto3.resource(
            's3',
            aws_access_key_id=self._get_env(self.aws_access_key_id),
            aws_secret_access_key=self._get_env(self.aws_secret_access_key),
            endpoint_url=self._get_env(self.endpoint_env_key),
            config=Config(s3={'addressing_style': 'path'})
        )

    # ----------------- Bucket operations -----------------

    def _create_bucket(self, bucket_name: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.create_bucket(Bucket=bucket_name)
            return f"✅ Bucket '{bucket_name}' created."
        except ClientError as e:
            return f"❌ Create bucket failed: {e}"

    def _delete_bucket(self, bucket_name: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.delete_bucket(Bucket=bucket_name)
            return f"🗑️ Bucket '{bucket_name}' deleted."
        except ClientError as e:
            return f"❌ Delete bucket failed: {e}"

    def _list_buckets(self) -> str:
        s3 = self._get_s3_client()
        try:
            buckets = s3.list_buckets()
            if not buckets["Buckets"]:
                return "📦 No buckets found."
            return "\n".join([f"📁 {b['Name']}" for b in buckets["Buckets"]])
        except ClientError as e:
            return f"❌ List buckets failed: {e}"

    # ----------------- Object operations -----------------

    def _upload_file(self, bucket_name: str, file_path: str) -> str:
        s3 = self._get_s3_resource()
        object_key = os.path.basename(file_path)
        try:
            bucket = s3.Bucket(bucket_name)
            obj = bucket.Object(object_key)
            with open(file_path, 'rb') as f:
                obj.put(Body=f)
            obj.wait_until_exists()
            return f"✅ Uploaded '{object_key}' to '{bucket_name}'"
        except ClientError as e:
            return f"❌ Upload failed: {e}"

    def _put_object(self, bucket_name: str, object_key: str, body: bytes) -> str:
        s3 = self._get_s3_client()
        try:
            s3.put_object(Bucket=bucket_name, Key=object_key, Body=body)
            return f"✅ Put object '{object_key}' to '{bucket_name}'"
        except ClientError as e:
            return f"❌ Put object failed: {e}"

    def _get_object(self, bucket_name: str, object_key: str) -> bytes:
        s3 = self._get_s3_client()
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()
        except ClientError as e:
            raise RuntimeError(f"❌ Get object failed: {e}")

    def _download_file(self, bucket_name: str, object_key: str, download_path: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.download_file(bucket_name, object_key, download_path)
            return f"✅ Object '{object_key}' downloaded to '{download_path}'."
        except ClientError as e:
            return f"❌ Download failed: {e}"

    def _head_object(self, bucket_name: str, object_key: str) -> dict:
        s3 = self._get_s3_client()
        try:
            return s3.head_object(Bucket=bucket_name, Key=object_key)
        except ClientError as e:
            raise RuntimeError(f"❌ Head object failed: {e}")

    def _delete_object(self, bucket_name: str, object_key: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.delete_object(Bucket=bucket_name, Key=object_key)
            return f"🗑️ Deleted object '{object_key}' from '{bucket_name}'"
        except ClientError as e:
            return f"❌ Delete object failed: {e}"

    def _delete_objects(self, bucket_name: str, object_keys: list) -> str:
        s3 = self._get_s3_client()
        try:
            objects = [{'Key': key} for key in object_keys]
            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
            return f"🗑️ Deleted {len(object_keys)} objects from '{bucket_name}'"
        except ClientError as e:
            return f"❌ Delete multiple objects failed: {e}"

    def _copy_object(self, source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> str:
        s3 = self._get_s3_client()
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            s3.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
            return f"📄 Copied '{source_key}' to '{dest_key}' in bucket '{dest_bucket}'"
        except ClientError as e:
            return f"❌ Copy failed: {e}"

    def _list_objects(self, bucket_name: str, prefix: str = "") -> list:
        s3 = self._get_s3_client()
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            objects = response.get("Contents", [])
            return [f"• {obj['Key']} (Size: {obj['Size']})" for obj in objects]
        except ClientError as e:
            raise RuntimeError(f"❌ List objects failed: {e}")

    def _generate_presigned_url(self, bucket_name: str, object_key: str, expires_in: int = 3600,
                                method: str = "get_object") -> str:
        s3 = self._get_s3_client()
        try:
            url = s3.generate_presigned_url(
                method,
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            return f"❌ Presigned URL failed: {e}"

    # ----------------- Multipart Upload -----------------

    def _create_multipart_upload(self, bucket_name: str, object_key: str) -> str:
        s3 = self._get_s3_client()
        try:
            response = s3.create_multipart_upload(Bucket=bucket_name, Key=object_key)
            return response["UploadId"]
        except ClientError as e:
            raise RuntimeError(f"❌ Create multipart upload failed: {e}")

    def _upload_part(self, bucket_name: str, object_key: str, upload_id: str, part_number: int, data: bytes) -> dict:
        s3 = self._get_s3_client()
        try:
            response = s3.upload_part(
                Bucket=bucket_name,
                Key=object_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=data
            )
            return {
                "PartNumber": part_number,
                "ETag": response["ETag"]
            }
        except ClientError as e:
            raise RuntimeError(f"❌ Upload part failed: {e}")

    def _complete_multipart_upload(self, bucket_name: str, object_key: str, upload_id: str, parts: list) -> str:
        s3 = self._get_s3_client()
        try:
            s3.complete_multipart_upload(
                Bucket=bucket_name,
                Key=object_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )
            return f"✅ Multipart upload complete for '{object_key}'"
        except ClientError as e:
            raise RuntimeError(f"❌ Complete multipart failed: {e}")

    def _abort_multipart_upload(self, bucket_name: str, object_key: str, upload_id: str) -> str:
        s3 = self._get_s3_client()
        try:
            s3.abort_multipart_upload(
                Bucket=bucket_name,
                Key=object_key,
                UploadId=upload_id
            )
            return f"🚫 Multipart upload aborted for '{object_key}'"
        except ClientError as e:
            raise RuntimeError(f"❌ Abort multipart upload failed: {e}")
