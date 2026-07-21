"""
Built-in Visual Template Packs for layout, composition, aspect ratios, and prompt compilation rules.
"""

from typing import List
from services.visual_engine.plugin_system import BaseTemplatePack, PluginRegistry
from services.visual_engine.schemas import SceneGraph, CameraSpec, LightingSpec, StyleSpec, LayoutSpec


class CommercialPhotographyTemplate(BaseTemplatePack):
    @property
    def template_name(self) -> str:
        return "commercial_photography"

    def apply_layout(self, scene_graph: SceneGraph, aspect_ratio: str = "16:9") -> LayoutSpec:
        return LayoutSpec(
            copy_space="clean open negative space in upper left quadrant for typography overlay",
            visual_hierarchy="primary focus on main subjects in right two-thirds rule-of-thirds frame, background environment softly blurred",
            aspect_ratio=aspect_ratio
        )

    def format_prompt(
        self,
        scene_graph: SceneGraph,
        camera: CameraSpec,
        lighting: LightingSpec,
        style: StyleSpec,
        layout: LayoutSpec
    ) -> str:
        # Build clean visual elements without narrative fluff
        node_descriptions = []
        for node in scene_graph.nodes:
            attrs_str = ", ".join(f"{k}: {v}" for k, v in node.attributes.items() if v)
            if attrs_str:
                node_descriptions.append(f"{node.name} [{node.spatial_zone}] ({attrs_str})")
            else:
                node_descriptions.append(f"{node.name} [{node.spatial_zone}]")

        edges_descriptions = []
        for edge in scene_graph.edges:
            edges_descriptions.append(f"{edge.source_id} {edge.relation.replace('_', ' ')} {edge.target_id}")

        prompt_parts = [
            f"Professional {style.rendering_style} capturing key climax scene: {scene_graph.climax_moment_summary}.",
            f"COMPOSITION: {camera.shot_type}, {camera.camera_angle}, {camera.lens}, {camera.perspective}. {layout.visual_hierarchy}. {layout.copy_space}.",
            f"MAIN SUBJECTS AND SPATIAL LAYOUT: {'; '.join(node_descriptions)}.",
            f"RELATIONSHIPS AND INTERACTIONS: {'; '.join(edges_descriptions)}.",
            f"ENVIRONMENT & LIGHTING: {scene_graph.environment_summary}. {lighting.primary_light}, {lighting.color_temperature}, {lighting.shadows}.",
            f"ATMOSPHERE & PALETTE: {lighting.mood}. Mood: {scene_graph.mood_and_atmosphere}. Color palette: {', '.join(style.color_palette)}. Textures: {', '.join(style.materials_and_textures)}.",
            f"SPECIFICATIONS: Shot on 35mm sensor, ultra-realistic detail, crisp focus, shallow depth of field, 8k resolution, photorealistic."
        ]
        return " ".join(prompt_parts)


class InfographicIllustrationTemplate(BaseTemplatePack):
    @property
    def template_name(self) -> str:
        return "infographic_illustration"

    def apply_layout(self, scene_graph: SceneGraph, aspect_ratio: str = "16:9") -> LayoutSpec:
        return LayoutSpec(
            copy_space="generous dedicated white space at top for infographic headline and icon callouts",
            visual_hierarchy="central bold vector illustration of key climax moment, surrounding icon markers for key steps",
            aspect_ratio=aspect_ratio
        )

    def format_prompt(
        self,
        scene_graph: SceneGraph,
        camera: CameraSpec,
        lighting: LightingSpec,
        style: StyleSpec,
        layout: LayoutSpec
    ) -> str:
        nodes_summary = ", ".join(n.name for n in scene_graph.nodes)
        prompt_parts = [
            f"Modern 3D isometric infographic illustration representing: {scene_graph.climax_moment_summary}.",
            f"VISUAL ELEMENTS: {nodes_summary}.",
            f"LAYOUT: {layout.visual_hierarchy}. {layout.copy_space}.",
            f"STYLE: Clean vector iconography, smooth gradients, soft contact ambient occlusion shadows, vibrant color palette: {', '.join(style.color_palette)}.",
            f"ENVIRONMENT: Minimalist clean isometric backdrop of {scene_graph.environment_summary}.",
            f"QUALITY: High clarity enterprise presentation graphics, Behance trending style, 8k resolution."
        ]
        return " ".join(prompt_parts)


class CinematicStorytellingTemplate(BaseTemplatePack):
    @property
    def template_name(self) -> str:
        return "cinematic_storytelling"

    def apply_layout(self, scene_graph: SceneGraph, aspect_ratio: str = "16:9") -> LayoutSpec:
        return LayoutSpec(
            copy_space="cinematic letterbox format with dark atmospheric negative space on left side",
            visual_hierarchy="dramatic focal point on human emotion and central conflict in rule of thirds",
            aspect_ratio=aspect_ratio
        )

    def format_prompt(
        self,
        scene_graph: SceneGraph,
        camera: CameraSpec,
        lighting: LightingSpec,
        style: StyleSpec,
        layout: LayoutSpec
    ) -> str:
        prompt_parts = [
            f"Cinematic wide film still depicting {scene_graph.climax_moment_summary}.",
            f"CAMERA & OPTICS: Panavision anamorphic lens, 35mm film grain, shallow depth of field, golden hour lighting.",
            f"SCENE SETUP: {scene_graph.environment_summary}. Environment lighting: {lighting.primary_light}.",
            f"ATMOSPHERE: Intense emotional mood ({scene_graph.mood_and_atmosphere}), dramatic volumetric light rays, atmospheric haze.",
            f"COLOR GRADE: Teal and warm orange split color grading, high contrast filmic tone curve."
        ]
        return " ".join(prompt_parts)


# Register built-in templates
PluginRegistry.register_template(CommercialPhotographyTemplate())
PluginRegistry.register_template(InfographicIllustrationTemplate())
PluginRegistry.register_template(CinematicStorytellingTemplate())
