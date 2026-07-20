"""Unit tests for S3 path metadata parsing and preservation."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.s3_storage import parse_s3_object_path


def test_parse_nested_s3_key():
    parsed = parse_s3_object_path(
        "Warehouse/Safety/Heat Stress Prevention Guide.pdf"
    )
    assert parsed["filename"] == "Heat Stress Prevention Guide.pdf"
    assert parsed["folder"] == "Warehouse"
    assert parsed["subfolder"] == "Safety"
    assert parsed["s3_key"] == "Warehouse/Safety/Heat Stress Prevention Guide.pdf"
    assert parsed["folder_path"] == "Warehouse/Safety"


def test_parse_deep_nested_s3_key():
    parsed = parse_s3_object_path("Warehouse/Safety/Heat Stress/Guide.pdf")
    assert parsed["filename"] == "Guide.pdf"
    assert parsed["folder"] == "Warehouse"
    assert parsed["subfolder"] == "Safety/Heat Stress"
    assert parsed["folder_path"] == "Warehouse/Safety/Heat Stress"


def test_parse_preserves_spaces_and_casing():
    key = "HR/Policies/Leave Policy.pdf"
    parsed = parse_s3_object_path(key)
    assert parsed["filename"] == "Leave Policy.pdf"
    assert parsed["folder"] == "HR"
    assert parsed["subfolder"] == "Policies"
    assert parsed["s3_key"] == key


def test_parse_root_file():
    parsed = parse_s3_object_path("standalone.docx")
    assert parsed["filename"] == "standalone.docx"
    assert parsed["folder"] == ""
    assert parsed["subfolder"] == ""
    assert parsed["s3_key"] == "standalone.docx"
