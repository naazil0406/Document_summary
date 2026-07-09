"""
AWS S3 storage for uploaded documents.

The parsers (PDFParser / DocxParser / ExcelParser) all need a real local
file path — fitz, python-docx, and openpyxl each open files off disk, and
the OCR path in pdf_parser.py rasterises pages via a local Docker mount.
Rewriting all three parsers to stream bytes straight from S3 would be a
much larger change for little practical benefit, since Streamlit's own
working directory is already local disk.

So the model here is simple and matches what was asked for:

    upload  -> saved locally (as before, so parsing/ingestion works
               unchanged) AND mirrored up to S3 as the durable copy
    fetch   -> anything present in S3 but missing from the local
               PDF_FOLDER (e.g. a fresh checkout, a restarted container,
               a second machine) gets downloaded back down on demand

S3 is the source of truth for "what documents exist"; the local folder is
a working cache of it.

This module is optional: if S3_BUCKET_NAME isn't set in .env, app.py's
get_s3_storage() returns None and the app runs exactly as it did before —
local-folder-only, no AWS calls, no boto3 credentials required.
"""

import logging
import os
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Storage:
    """Thin wrapper around an S3 bucket used as durable document storage."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: str = "us-east-1",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
    ):
        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME is required to use S3Storage.")

        self.bucket_name = bucket_name
        # Normalize so keys never end up with a double/missing slash.
        self.prefix = prefix.strip("/")
        if self.prefix:
            self.prefix += "/"

        # If explicit keys are provided in .env, use them. Otherwise fall
        # back to boto3's normal credential chain (env vars, ~/.aws/credentials,
        # an IAM role if running on EC2/ECS/Lambda, etc.) — this lets the app
        # work in AWS-hosted environments without ever putting a key in .env.
        client_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("s3", **client_kwargs)

    def _key_for(self, filename: str) -> str:
        return f"{self.prefix}{filename}"

    def upload_file(self, local_path: str, filename: str) -> str:
        """Upload a local file to S3. Returns the s3:// URI it was stored at."""
        key = self._key_for(filename)
        try:
            self.client.upload_file(local_path, self.bucket_name, key)
        except NoCredentialsError as exc:
            raise RuntimeError(
                "No AWS credentials found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "in .env, or configure credentials via the AWS CLI / an IAM role."
            ) from exc
        except ClientError as exc:
            raise RuntimeError(f"Failed to upload '{filename}' to S3: {exc}") from exc

        uri = f"s3://{self.bucket_name}/{key}"
        logger.info("Uploaded '%s' to %s.", filename, uri)
        return uri

    def download_file(self, filename: str, local_path: str) -> bool:
        """Download filename from S3 to local_path. Returns True if downloaded,
        False if the object doesn't exist in the bucket."""
        key = self._key_for(filename)
        try:
            self.client.download_file(self.bucket_name, key, local_path)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise RuntimeError(f"Failed to download '{filename}' from S3: {exc}") from exc

        logger.info("Downloaded '%s' from S3 to '%s'.", filename, local_path)
        return True

    def file_exists(self, filename: str) -> bool:
        key = self._key_for(filename)
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise RuntimeError(f"Failed to check '{filename}' in S3: {exc}") from exc

    def list_files(self) -> List[str]:
        """Return the basenames of every object stored under prefix in the bucket."""
        filenames: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key == self.prefix:  # the "folder" marker object itself, if any
                        continue
                    filenames.append(os.path.basename(key))
        except ClientError as exc:
            raise RuntimeError(f"Failed to list objects in S3 bucket '{self.bucket_name}': {exc}") from exc

        return sorted(filenames)

    def rename_file(self, old_filename: str, new_filename: str) -> None:
        """Rename an object in S3 by copying it to the new key and deleting
        the old one (S3 has no atomic rename). Used by the canonical naming
        migration; no-op if old_filename == new_filename."""
        if old_filename == new_filename:
            return
        old_key = self._key_for(old_filename)
        new_key = self._key_for(new_filename)
        try:
            self.client.copy_object(
                Bucket=self.bucket_name,
                CopySource={"Bucket": self.bucket_name, "Key": old_key},
                Key=new_key,
            )
            self.client.delete_object(Bucket=self.bucket_name, Key=old_key)
        except ClientError as exc:
            raise RuntimeError(
                f"Failed to rename '{old_filename}' -> '{new_filename}' in S3: {exc}"
            ) from exc
        logger.info("Renamed S3 object '%s' -> '%s'.", old_filename, new_filename)

    def sync_down(self, local_folder: str) -> List[str]:
        """Download every S3 object not already present locally into
        local_folder. Returns the list of filenames actually downloaded."""
        os.makedirs(local_folder, exist_ok=True)
        downloaded: List[str] = []

        for filename in self.list_files():
            local_path = os.path.join(local_folder, filename)
            if os.path.isfile(local_path):
                continue
            if self.download_file(filename, local_path):
                downloaded.append(filename)

        return downloaded