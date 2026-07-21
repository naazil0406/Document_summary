"""
Standalone S3 diagnostic — bypasses the app entirely so you can confirm
whether the bucket/prefix actually has the files you expect, independent
of any app caching.

Usage:
    python check_s3_prefix.py

Reads the same AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION /
S3_BUCKET_NAME / S3_PREFIX values from your .env (via python-dotenv), so
it mirrors exactly what the app would use.
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

bucket = os.getenv("S3_BUCKET_NAME", "")
prefix = os.getenv("S3_PREFIX", "").strip().strip("/")
prefix = f"{prefix}/" if prefix else ""
region = os.getenv("AWS_REGION", "us-east-1")

print(f"Bucket: {bucket!r}")
print(f"Prefix being used: {prefix!r}")
print(f"Region: {region!r}")
print("-" * 60)

client = boto3.client(
    "s3",
    region_name=region,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
)

# 1. Confirm the bucket is reachable at all.
try:
    client.head_bucket(Bucket=bucket)
    print("Bucket reachable: YES")
except Exception as exc:
    print(f"Bucket reachable: NO -> {exc}")
    raise SystemExit(1)

# 2. List everything under the configured prefix, recursively, exactly
#    like s3_storage.py's list_objects() does (no Delimiter).
paginator = client.get_paginator("list_objects_v2")
count = 0
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        print(f"  {obj['Key']}")
        count += 1

print("-" * 60)
print(f"Total objects found under prefix {prefix!r}: {count}")

if count == 0:
    print(
        "\nNo objects found under that exact prefix. Double-check the "
        "folder name's exact case and spelling as it appears in the S3 "
        "console — S3 keys are case-sensitive."
    )