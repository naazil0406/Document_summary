"""
Unit & Integration tests for new parsers and pipeline flow (Markdown, XML, JSON, Transcript, Re-ranker).
"""

import os
import sys
import tempfile
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.markdown_parser import MarkdownParser
from services.xml_parser import XMLParser
from services.json_parser import JSONParser
from services.transcript_parser import TranscriptParser
from services.semantic_boundary_detector import SemanticBoundaryDetector
from services.reranker import ReRankerService


def test_markdown_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Heading 1\n\nThis is a test markdown paragraph.\n\n| Col1 | Col2 |\n| --- | --- |\n| Val1 | Val2 |\n")
        f_path = f.name

    try:
        parser = MarkdownParser()
        pages = parser.extract_pages(f_path)
        assert len(pages) == 1
        assert "Heading 1" in pages[0].text
        assert "Val1" in pages[0].text
    finally:
        os.remove(f_path)


def test_xml_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write("<root><title>Safety Guide</title><section><p>Warning: Hot Surfaces</p></section></root>")
        f_path = f.name

    try:
        parser = XMLParser()
        pages = parser.extract_pages(f_path)
        assert len(pages) == 1
        assert "Safety Guide" in pages[0].text
        assert "Hot Surfaces" in pages[0].text
    finally:
        os.remove(f_path)


def test_json_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"title": "Policy Document", "steps": ["Step 1: Inspect gear", "Step 2: Wear helmet"]}')
        f_path = f.name

    try:
        parser = JSONParser()
        pages = parser.extract_pages(f_path)
        assert len(pages) == 1
        assert "Policy Document" in pages[0].text
        assert "Inspect gear" in pages[0].text
    finally:
        os.remove(f_path)


def test_transcript_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("[00:01:15] Speaker A: Please follow safety procedure A1.\n[00:01:45] Speaker B: Understood.")
        f_path = f.name

    try:
        parser = TranscriptParser()
        pages = parser.extract_pages(f_path)
        assert len(pages) == 1
        assert "Speaker A" in pages[0].text
        assert "safety procedure A1" in pages[0].text
    finally:
        os.remove(f_path)


def test_reranker_pass_through_when_model_unloaded():
    reranker = ReRankerService(model_name="non_existent_model_xyz", device="cpu")
    chunks = [
        {"chunk_id": "c1", "text": "Safety procedures for emergency evacuation", "score": 0.5},
        {"chunk_id": "c2", "text": "Quarterly financial report overview", "score": 0.3},
    ]
    result = reranker.rerank("emergency evacuation", chunks, top_k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"
