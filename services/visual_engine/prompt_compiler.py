"""
Universal Prompt Compiler module.
Receives ONLY the Scene Graph + Plugins to compile professional image prompt specs.
Never accesses user raw query directly.
"""

import logging
from typing import List, Optional

from services.visual_engine.schemas import (
    SceneGraph,
    CompiledPromptSpec,
    CameraSpec,
    LightingSpec,
    StyleSpec,
    LayoutSpec,
)
from services.visual_engine.plugin_system import PluginRegistry

logger = logging.getLogger(__name__)


class UniversalPromptCompiler:
    """Converts a SceneGraph into a production-grade compiled image prompt specification."""

    def compile_prompt(
        self,
        scene_graph: SceneGraph,
        domain_name: Optional[str] = None,
        template_name: Optional[str] = None,
        aspect_ratio: str = "16:9"
    ) -> CompiledPromptSpec:
        applied_rules: List[str] = ["SceneGraph-Only Compilation"]

        # Resolve Domain Pack
        d_name = domain_name or scene_graph.domain_type or "warehouse"
        domain_pack = PluginRegistry.get_domain(d_name)
        if domain_pack:
            scene_graph = domain_pack.enhance_scene_graph(scene_graph)
            camera = domain_pack.get_camera_preset()
            lighting = domain_pack.get_lighting_preset()
            style = domain_pack.get_style_preset()
            domain_negatives = domain_pack.get_domain_negative_prompts()
            applied_rules.append(f"Domain Pack: {domain_pack.domain_name}")
        else:
            camera = CameraSpec()
            lighting = LightingSpec()
            style = StyleSpec()
            domain_negatives = []

        # Resolve Template Pack
        t_name = template_name or "commercial_photography"
        template_pack = PluginRegistry.get_template(t_name)
        if template_pack:
            layout = template_pack.apply_layout(scene_graph, aspect_ratio=aspect_ratio)
            positive_prompt = template_pack.format_prompt(
                scene_graph=scene_graph,
                camera=camera,
                lighting=lighting,
                style=style,
                layout=layout
            )
            applied_rules.append(f"Template Pack: {template_pack.template_name}")
        else:
            layout = LayoutSpec(aspect_ratio=aspect_ratio)
            positive_prompt = f"Professional photograph of {scene_graph.climax_moment_summary} in {scene_graph.environment_summary}."

        # Compile Negative Prompt
        base_negatives = [
            "blurry", "out of focus", "low quality", "distorted anatomy",
            "extra fingers", "mutated limbs", "overexposed", "underexposed",
            "watermark", "signature", "text errors", "garbled letters"
        ]
        all_negatives = list(dict.fromkeys(base_negatives + domain_negatives))
        negative_prompt = ", ".join(all_negatives)

        return CompiledPromptSpec(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            camera=camera,
            lighting=lighting,
            style=style,
            layout=layout,
            applied_rules=applied_rules
        )
