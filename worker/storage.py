
"""S3 input/output for OpenDock Worker."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


MAX_MSA_BYTES = 100 * 1024 * 1024


def parse_s3_uri(
    uri: str,
    *,
    is_prefix: bool = False,
) -> tuple[str, str]:

    if (
        not isinstance(uri, str)
        or not uri.startswith("s3://")
    ):
        raise ValueError(
            "Expected s3:// URI"
        )

    parts = urlsplit(uri)
    key = parts.path[1:]

    if (
        parts.scheme != "s3"
        or not parts.netloc
        or not key
        or parts.query
        or parts.fragment
        or key.startswith("/")
        or "\\" in key
        or "%" in key
        or "//" in key
        or any(
            p in (".", "..")
            for p in key.split("/")
        )
        or key.endswith("/") != is_prefix
    ):
        raise ValueError(
            "Invalid S3 URI"
        )

    return parts.netloc, key


def _check_bucket(bucket: str) -> None:

    allowed = os.environ.get(
        "OPENDOCK_S3_BUCKET"
    )

    if not allowed or bucket != allowed:
        raise ValueError(
            "S3 bucket is not authorized"
        )


def s3_client():

    import boto3

    return boto3.client("s3")


def download_msa(
    uri: str,
    destination: Path,
) -> None:

    bucket, key = parse_s3_uri(uri)

    _check_bucket(bucket)

    if not key.endswith(".a3m"):
        raise ValueError(
            "MSA must be .a3m"
        )

    client = s3_client()

    size = client.head_object(
        Bucket=bucket,
        Key=key,
    )["ContentLength"]

    if not 0 < size <= MAX_MSA_BYTES:
        raise ValueError(
            "MSA is empty or too large"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    client.download_file(
        bucket,
        key,
        str(destination),
    )

    if not (
        0 < destination.stat().st_size
        <= MAX_MSA_BYTES
    ):
        raise ValueError(
            "Downloaded MSA size invalid"
        )


def upload_results(
    directory: Path,
    manifest: dict,
    output_uri: str,
) -> dict:
    """Upload artifacts, then manifest as completion marker."""

    import json

    bucket, prefix = parse_s3_uri(
        output_uri,
        is_prefix=True,
    )

    _check_bucket(bucket)

    client = s3_client()

    for item in manifest["files"]:

        path = directory / item["path"]

        key = (
            prefix
            + "artifacts/"
            + item["path"]
        )

        client.upload_file(
            str(path),
            bucket,
            key,
        )

        item["s3_uri"] = (
            f"s3://{bucket}/{key}"
        )

    manifest["manifest_s3_uri"] = (
        f"s3://{bucket}/{prefix}manifest.json"
    )

    client.put_object(
        Bucket=bucket,
        Key=prefix + "manifest.json",
        Body=json.dumps(
            manifest,
            ensure_ascii=False,
        ).encode("utf-8"),
        ContentType="application/json",
    )

    return manifest
