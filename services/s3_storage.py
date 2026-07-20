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

Multiple prefixes: S3_PREFIX can be a single folder ("data/") or a
comma-separated list of folders ("data/,July2026/") when documents are
split across more than one top-level "folder" in the same bucket. Every
read/write operation loops over all configured prefixes, so uploads land
under the first prefix (the primary/default one) while listing and
syncing pull from all of them.
"""

import logging
import os
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from services.canonical_naming import current_month_folder

logger = logging.getLogger(__name__)


def _normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().strip("/")
    return f"{prefix}/" if prefix else ""


class S3Storage:
    """Thin wrapper around an S3 bucket used as durable document storage.

    ``prefix`` accepts either a single folder ("data/") or a
    comma-separated list of folders ("data/,July2026/") to read from
    multiple "folders" in the same bucket.
    """

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

        # Normalize each comma-separated entry so keys never end up with a
        # double/missing slash. self.prefixes[0] is the "primary" prefix
        # new uploads are written under; every prefix is read from.
        raw_prefixes = [p for p in prefix.split(",") if p.strip()] or [""]
        self.prefixes: List[str] = [_normalize_prefix(p) for p in raw_prefixes]

        # Backward-compatible single-prefix attribute (primary/default one)
        # for any older code that still reads .prefix directly.
        self.prefix = self.prefixes[0]

        # If explicit keys are provided in .env, use them. Otherwise fall
        # back to boto3's normal credential chain (env vars, ~/.aws/credentials,
        # an IAM role if running on EC2/ECS/Lambda, etc.) — this lets the app
        # work in AWS-hosted environments without ever putting a key in .env.
        client_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("s3", **client_kwargs)

    def _key_for(self, filename: str, prefix: Optional[str] = None) -> str:
        prefix = self.prefix if prefix is None else prefix
        return f"{prefix}{filename}"

    def _discover_root_prefixes(self) -> List[str]:
        """List every top-level "folder" that actually exists in the
        bucket (e.g. "july/", "august/"), so monthly folders already
        present in the bucket are always searched -- no need to list each
        one in S3_PREFIX by hand."""
        prefixes: List[str] = []
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Delimiter="/"):
                for common in page.get("CommonPrefixes", []):
                    prefixes.append(common["Prefix"])
        except ClientError as exc:
            logger.warning("Could not auto-discover S3 folders in '%s': %s", self.bucket_name, exc)
        return prefixes

    def _all_prefixes(self) -> List[str]:
        """Configured prefixes plus every folder discovered at the bucket
        root, deduplicated, configured ones first."""
        merged: List[str] = []
        for p in self.prefixes + self._discover_root_prefixes():
            if p not in merged:
                merged.append(p)
        return merged

    def upload_file(self, local_path: str, filename: str) -> str:
        """Upload a local file to S3 under the current month's folder
        (e.g. "july/<filename>"), auto-detected from today's date to match
        the bucket's existing monthly-folder layout. Returns the s3://
        URI it was stored at."""
        month_prefix = _normalize_prefix(current_month_folder())
        key = self._key_for(filename, prefix=month_prefix)
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

    def _find_key(self, filename: str) -> Optional[str]:
        """Return the first existing key for filename across all configured
        AND auto-discovered (e.g. monthly) prefixes, or None if it doesn't
        exist under any of them."""
        for prefix in self._all_prefixes():
            key = self._key_for(filename, prefix)
            try:
                self.client.head_object(Bucket=self.bucket_name, Key=key)
                return key
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code in ("404", "NoSuchKey"):
                    continue
                raise RuntimeError(f"Failed to check '{filename}' in S3: {exc}") from exc
        return None

    def download_file(self, filename: str, local_path: str) -> bool:
        """Download filename from S3 (searched across all configured
        prefixes) to local_path. Returns True if downloaded, False if the
        object doesn't exist in any configured prefix."""
        key = self._find_key(filename)
        if key is None:
            return False
        try:
            self.client.download_file(self.bucket_name, key, local_path)
        except ClientError as exc:
            raise RuntimeError(f"Failed to download '{filename}' from S3: {exc}") from exc

        logger.info("Downloaded '%s' from S3 to '%s'.", filename, local_path)
        return True

    def file_exists(self, filename: str) -> bool:
        return self._find_key(filename) is not None

    def list_files(self) -> List[str]:
        """Return the basenames of every object stored under any configured
        prefix in the bucket (deduplicated, sorted)."""
        filenames: set[str] = set()
        paginator = self.client.get_paginator("list_objects_v2")
        for prefix in self._all_prefixes():
            try:
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if key == prefix:  # the "folder" marker object itself, if any
                            continue
                        filenames.add(os.path.basename(key))
            except ClientError as exc:
                raise RuntimeError(
                    f"Failed to list objects in S3 bucket '{self.bucket_name}' "
                    f"under prefix '{prefix}': {exc}"
                ) from exc

        return sorted(filenames)

    def list_objects(self) -> List[dict]:
        """Return authoritative S3 objects with their complete keys.

        Unlike :meth:`list_files`, this preserves folder information and is
        therefore safe when two folders contain the same basename.
        """
        objects: dict[str, dict] = {}
        paginator = self.client.get_paginator("list_objects_v2")
        for prefix in self._all_prefixes():
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key == prefix or key.endswith("/"):
                        continue
                    objects[key] = {
                        "key": key,
                        "filename": os.path.basename(key),
                        "folder_name": os.path.dirname(key).strip("/"),
                        "upload_date": obj.get("LastModified").isoformat() if obj.get("LastModified") else "",
                    }
        return [objects[key] for key in sorted(objects)]

    def rename_file(self, old_filename: str, new_filename: str) -> None:
        """Rename an object in S3 by copying it to the new key and deleting
        the old one (S3 has no atomic rename). Used by the canonical naming
        migration; no-op if old_filename == new_filename.

        Renames in place under whichever configured prefix the file was
        actually found in."""
        if old_filename == new_filename:
            return
        old_key = self._find_key(old_filename)
        if old_key is None:
            logger.warning(
                "Could not rename '%s' -> '%s' in S3: not found under any configured prefix.",
                old_filename, new_filename,
            )
            return
        prefix = old_key[: len(old_key) - len(old_filename)]
        new_key = self._key_for(new_filename, prefix)
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
        """Download every S3 object (across all configured + discovered
        prefixes) not already present locally into local_folder, mirroring
        whatever month subfolder it lives under in the bucket (e.g. an
        object at "july/<file>" lands at "<local_folder>/july/<file>").
        Returns the list of filenames actually downloaded."""
        os.makedirs(local_folder, exist_ok=True)
        downloaded: List[str] = []

        for obj in self.list_objects():
            filename, key = obj["filename"], obj["key"]
            subfolder = obj["folder_name"]
            local_dir = os.path.join(local_folder, subfolder) if subfolder else local_folder
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                continue
            try:
                self.client.download_file(self.bucket_name, key, local_path)
            except ClientError as exc:
                raise RuntimeError(f"Failed to download '{filename}' from S3: {exc}") from exc
            downloaded.append(filename)

        return downloaded
