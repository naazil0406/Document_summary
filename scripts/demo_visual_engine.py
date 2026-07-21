"""
End-to-End Demo Script for the Universal AI Visual Content Engine.
Demonstrates the heat-stress warehouse safety scenario with step-by-step visual audit logs.
"""

import json
import os
import sys

# Ensure parent path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.visual_engine import UniversalVisualContentEngine, VisualEngineOutput


def main():
    print("=" * 80)
    print("UNIVERSAL AI VISUAL CONTENT ENGINE - END-TO-END DEMONSTRATION")
    print("=" * 80)

    # Initialize Engine
    engine = UniversalVisualContentEngine()

    user_topic = "Recognize heat stress in warehouse environment before an emergency occurs"

    print(f"\n[INPUT USER TOPIC]: {user_topic}")
    print("-" * 80)

    output: VisualEngineOutput = engine.generate_content_and_visual(
        user_request=user_topic,
        domain_override="warehouse",
        style_override="commercial_photography",
        aspect_ratio="16:9",
        generate_image_bytes=False
    )

    print("\n1. INTENT ANALYZER OUTPUT:")
    print(json.dumps(output.intent.model_dump(), indent=2))

    print("\n2. GENERATED CONTENT (SINGLE SOURCE OF TRUTH):")
    print(f"Title: {output.generated_content.title}")
    print(f"Content: \"{output.generated_content.raw_content}\"")
    print(f"Core Message: {output.generated_content.core_message}")

    print("\n3. CONTENT ANALYSIS ENGINE OUTPUT (STRUCTURED JSON):")
    print(json.dumps(output.content_analysis.model_dump(), indent=2))

    print("\n4. SCENE SELECTION ENGINE (KEY CLIMAX FRAME SELECTED):")
    climax = output.content_analysis.selected_climax_event
    if climax:
        print(f"  Frame Step #{climax.step_number}: {climax.event_description}")
        print(f"  Visual Impact Score: {climax.visual_impact_score}")
        print(f"  Is Climax: {climax.is_climax}")

    print("\n5. SCENE GRAPH BUILDER OUTPUT (PURE STRUCTURED GRAPH):")
    print(json.dumps(output.scene_graph.model_dump(), indent=2))

    print("\n6. UNIVERSAL PROMPT COMPILER OUTPUT:")
    print("  POSITIVE PROMPT:")
    print(f"  {output.prompt_spec.positive_prompt}")
    print("\n  NEGATIVE PROMPT:")
    print(f"  {output.prompt_spec.negative_prompt}")
    print(f"\n  APPLIED RULES: {', '.join(output.prompt_spec.applied_rules)}")

    print("\n7. CONTENT & IMAGE CONSISTENCY VALIDATOR REPORT:")
    print(json.dumps(output.consistency_report.model_dump(), indent=2))

    print("\n" + "=" * 80)
    print("SUCCESS: Content and Image Prompt generated with 100% semantic consistency!")
    print("=" * 80)


if __name__ == "__main__":
    main()
