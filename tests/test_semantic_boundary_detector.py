"""Unit tests for the Semantic Boundary Detection engine."""

import json

from services.semantic_boundary_detector import (
    SemanticBoundaryDetector,
    detect_semantic_boundaries,
)


SAMPLE_DOCUMENT = {
    "document_title": "Heat Stress Prevention Guide",
    "document_type": "safety_guide",
    "sections": [
        {
            "section_id": "section_01",
            "title": "Heat Stress",
            "content": [],
            "subsections": [
                {
                    "section_id": "section_01_a",
                    "title": "Symptoms",
                    "content": [
                        "Heat exhaustion may include heavy sweating, weakness, and nausea.",
                        "Seek medical attention if symptoms persist.",
                    ],
                    "subsections": [],
                    "metadata": {"page_start": 18, "page_end": 19},
                },
                {
                    "section_id": "section_01_b",
                    "title": "Emergency Procedure",
                    "content": [
                        "Step 1: Move the person to a shaded area.\n"
                        "Step 2: Provide cool water.\n"
                        "Step 3: Call emergency services if needed."
                    ],
                    "subsections": [],
                    "metadata": {"page_start": 20, "page_end": 21},
                },
            ],
            "metadata": {},
        }
    ],
    "metadata": {
        "filename": "Heat Stress Prevention Guide.pdf",
        "folder": "Warehouse",
        "subfolder": "Safety",
        "s3_key": "knowledge-base/Warehouse/Heat Stress Prevention Guide.pdf",
    },
}


def test_detect_semantic_boundaries_basic_structure():
    result = detect_semantic_boundaries(SAMPLE_DOCUMENT)
    blocks = result["semantic_blocks"]

    assert len(blocks) >= 3
    assert blocks[0]["block_id"] == "block_000001"
    assert blocks[0]["heading_path"] == [
        "Warehouse",
        "Heat Stress Prevention Guide",
        "Heat Stress",
        "Symptoms",
    ]
    assert blocks[0]["parent_heading"] == "Heat Stress"
    assert blocks[0]["section_id"] == "section_01_a"
    assert blocks[0]["page_start"] == 18
    assert blocks[0]["page_end"] == 19
    assert blocks[0]["content_type"] == "paragraph"


def test_procedure_block_is_protected():
    result = detect_semantic_boundaries(SAMPLE_DOCUMENT)
    procedure_blocks = [
        block
        for block in result["semantic_blocks"]
        if block["content_type"] == "procedure"
    ]

    assert len(procedure_blocks) == 1
    procedure = procedure_blocks[0]
    assert procedure["protected"] is True
    assert "Step 1" in procedure["content"]
    assert "Step 3" in procedure["content"]
    assert procedure["heading_path"][-1] == "Emergency Procedure"
    assert procedure["page_start"] == 20
    assert procedure["page_end"] == 21


def test_metadata_preserved():
    result = detect_semantic_boundaries(SAMPLE_DOCUMENT)
    metadata = result["semantic_blocks"][0]["metadata"]

    assert metadata["filename"] == "Heat Stress Prevention Guide.pdf"
    assert metadata["folder"] == "Warehouse"
    assert metadata["subfolder"] == "Safety"
    assert metadata["s3_key"] == "knowledge-base/Warehouse/Heat Stress Prevention Guide.pdf"


def test_content_preserved_exactly():
    result = detect_semantic_boundaries(SAMPLE_DOCUMENT)
    original_lines = [
        "Heat exhaustion may include heavy sweating, weakness, and nausea.",
        "Seek medical attention if symptoms persist.",
        "Step 1: Move the person to a shaded area.",
        "Step 2: Provide cool water.",
        "Step 3: Call emergency services if needed.",
    ]
    combined = "\n".join(block["content"] for block in result["semantic_blocks"])

    for line in original_lines:
        assert line in combined


def test_warning_table_and_faq_detection():
    document = {
        "document_title": "Safety Manual",
        "sections": [
            {
                "section_id": "sec_1",
                "title": "Hazards",
                "content": [
                    "WARNING: High voltage area.",
                    "Q: What PPE is required?\nA: Hard hat and gloves.",
                    "| Hazard | Level |\n| Electrical | High |",
                ],
                "subsections": [],
                "metadata": {"page_start": 5, "page_end": 6},
            }
        ],
        "metadata": {"filename": "Safety Manual.pdf", "folder": "Plant"},
    }

    result = detect_semantic_boundaries(document)
    types = {block["content_type"] for block in result["semantic_blocks"]}

    assert "warning" in types
    assert "faq" in types
    assert "table" in types

    warning = next(b for b in result["semantic_blocks"] if b["content_type"] == "warning")
    assert warning["protected"] is True
    assert warning["content"] == "WARNING: High voltage area."


def test_structured_content_item_with_explicit_type():
    document = {
        "document_title": "Guide",
        "sections": [
            {
                "section_id": "sec_code",
                "title": "Examples",
                "content": [
                    {
                        "content_type": "code_block",
                        "content": "def hello():\n    return 'world'",
                        "metadata": {"page_start": 10, "page_end": 10},
                    }
                ],
                "subsections": [],
                "metadata": {},
            }
        ],
        "metadata": {"filename": "Guide.pdf"},
    }

    result = detect_semantic_boundaries(document)
    block = result["semantic_blocks"][0]

    assert block["content_type"] == "code_block"
    assert block["protected"] is True
    assert "def hello():" in block["content"]


def test_semantic_blocks_to_document_chunks_adapter():
    try:
        from services.chunking import document_chunks_from_semantic_blocks
    except ImportError:
        return

    result = detect_semantic_boundaries(SAMPLE_DOCUMENT)
    doc_chunks = document_chunks_from_semantic_blocks(result)

    assert len(doc_chunks) >= 3
    assert doc_chunks[0].metadata["semantic_block_id"] == "block_000001"
    assert doc_chunks[0].metadata["protected"] is False
    assert any(chunk.metadata.get("protected") for chunk in doc_chunks)


def test_output_is_valid_json():
    detector = SemanticBoundaryDetector()
    payload = detector.detect_json(SAMPLE_DOCUMENT)
    parsed = json.loads(payload)

    assert "semantic_blocks" in parsed
    assert isinstance(parsed["semantic_blocks"], list)
