"""
Unit tests for the Universal AI Visual Content Engine.
"""

import pytest
from services.visual_engine import UniversalVisualContentEngine, PluginRegistry
from services.visual_engine.schemas import VisualEngineOutput
from services.visual_engine.domain_packs import WarehouseSafetyDomain, HealthcareDomain
from services.visual_engine.template_packs import CommercialPhotographyTemplate


def test_plugin_registry():
    assert "warehouse" in PluginRegistry.list_domains()
    assert "commercial_photography" in PluginRegistry.list_templates()

    domain = PluginRegistry.get_domain("warehouse")
    assert isinstance(domain, WarehouseSafetyDomain)

    template = PluginRegistry.get_template("commercial_photography")
    assert isinstance(template, CommercialPhotographyTemplate)


def test_visual_engine_warehouse_heat_stress():
    engine = UniversalVisualContentEngine(llm_service=None, retriever_service=None, image_service=None)
    
    user_query = "Warehouse safety warning on heat stress and worker exhaustion"
    output: VisualEngineOutput = engine.generate_content_and_visual(
        user_request=user_query,
        domain_override="warehouse",
        generate_image_bytes=False
    )

    # 1. Check Intent
    assert output.intent.domain == "warehouse"
    assert output.intent.communication_purpose == "safety"

    # 2. Check Content is single source of truth
    assert "sun" in output.generated_content.raw_content.lower() or "heat" in output.generated_content.raw_content.lower() or "alex" in output.generated_content.raw_content.lower()

    # 3. Check Content Analysis Engine extractions
    assert len(output.content_analysis.main_characters) >= 1
    assert output.content_analysis.environment.location != ""

    # 4. Check Scene Selection Engine selected climax event
    assert output.content_analysis.selected_climax_event is not None
    assert output.content_analysis.selected_climax_event.is_climax is True

    # 5. Check Scene Graph
    assert len(output.scene_graph.nodes) >= 3
    assert output.scene_graph.climax_moment_summary != ""

    # 6. Check Universal Prompt Compiler
    pos_prompt = output.prompt_spec.positive_prompt
    neg_prompt = output.prompt_spec.negative_prompt
    assert "COMPOSITION" in pos_prompt
    assert "SPATIAL LAYOUT" in pos_prompt
    assert "ENVIRONMENT" in pos_prompt
    assert "cartoon" in neg_prompt or "blurry" in neg_prompt

    # 7. Check Consistency Validator
    assert output.consistency_report.is_consistent is True
    assert output.consistency_report.overall_score >= 0.80


def test_visual_engine_healthcare():
    engine = UniversalVisualContentEngine(llm_service=None, retriever_service=None, image_service=None)
    output = engine.generate_content_and_visual(
        user_request="Patient consultation for clinical trial",
        domain_override="healthcare",
        generate_image_bytes=False
    )
    assert output.intent.domain == "healthcare"
    assert output.consistency_report.is_consistent is True


def test_visual_engine_content_type_parsing():
    engine = UniversalVisualContentEngine(llm_service=None, retriever_service=None, image_service=None)
    output_recall = engine.generate_content_and_visual(
        user_request="Recall Card: heatwave awareness",
        generate_image_bytes=False
    )
    assert output_recall.intent.content_type == "Recall Card"
    assert output_recall.intent.raw_request == "heatwave awareness"

def test_visual_engine_school_awareness_no_refusal():
    engine = UniversalVisualContentEngine(llm_service=None, retriever_service=None, image_service=None)
    output = engine.generate_content_and_visual(
        user_request="Infographic: school awareness",
        generate_image_bytes=False
    )
    assert output.intent.domain == "education"
    assert output.intent.content_type == "Infographic"
    assert "isn't any specific context" not in output.generated_content.raw_content.lower()
    assert "provided materials" not in output.generated_content.raw_content.lower()
    assert output.consistency_report.is_consistent is True


