"""
Built-in Domain Packs for enterprise verticals (Warehouse/Safety, Healthcare, Corporate, Tech, General).
"""

from typing import List
from services.visual_engine.plugin_system import BaseDomainPack, PluginRegistry
from services.visual_engine.schemas import SceneGraph, CameraSpec, LightingSpec, StyleSpec, SceneNode


class WarehouseSafetyDomain(BaseDomainPack):
    @property
    def domain_name(self) -> str:
        return "warehouse"

    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        # Guarantee safety gear and industrial props exist on relevant nodes
        has_subject = False
        for node in scene_graph.nodes:
            if node.node_type == "subject":
                has_subject = True
                node.attributes.setdefault("clothing", "high-visibility safety vest, work trousers, steel-toe boots")
                node.attributes.setdefault("headwear", "hard hat / safety helmet")
            elif node.node_type == "object":
                if "rack" in node.name.lower() or "pallet" in node.name.lower():
                    node.attributes.setdefault("material", "heavy-duty industrial steel")

        if not has_subject:
            # Inject a prominent foreground subject worker so FLUX never generates an empty scene
            scene_graph.nodes.insert(0, SceneNode(
                id="worker_1",
                name="Alex (warehouse technician)",
                node_type="subject",
                spatial_zone="foreground center-right",
                attributes={
                    "clothing": "yellow high-visibility safety vest, dark work trousers, steel-toe boots",
                    "action": "taking a brief Rate Your State (RYS) pause, holding a clipboard and assessing heat fatigue",
                    "emotion": "focused, mindful, sweating slightly from physical exertion",
                    "headwear": "hard hat / safety helmet"
                }
            ))
        return scene_graph

    def get_camera_preset(self) -> CameraSpec:
        return CameraSpec(
            shot_type="medium wide eye-level shot",
            camera_angle="slightly low angle to emphasize human drama and warehouse scale",
            lens="50mm f/1.8 prime lens for crisp subject isolation",
            perspective="dynamic three-quarter isometric perspective"
        )

    def get_lighting_preset(self) -> LightingSpec:
        return LightingSpec(
            primary_light="intense midday sun streaming through elevated skylights and bay doors",
            color_temperature="5500K hot golden daylight",
            shadows="harsh defined cast shadows emphasizing heat intensity",
            mood="high urgency, realistic industrial environment"
        )

    def get_style_preset(self) -> StyleSpec:
        return StyleSpec(
            rendering_style="authentic documentary photojournalism",
            medium="award-winning 35mm photography",
            materials_and_textures=["high-vis reflective tape", "weathered cardboard boxes", "polished industrial concrete"],
            color_palette=["safety orange", "industrial caution yellow", "warm golden sunbeams", "cool slate grey"]
        )

    def get_domain_negative_prompts(self) -> List[str]:
        return [
            "cartoon", "anime", "over-dramatized action movie CGI", "missing safety vest",
            "bare foot", "disfigured hands", "extra limbs", "garbled signs", "text errors"
        ]


class HealthcareDomain(BaseDomainPack):
    @property
    def domain_name(self) -> str:
        return "healthcare"

    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        for node in scene_graph.nodes:
            if node.node_type == "subject":
                node.attributes.setdefault("clothing", "clean medical scrubs or white physician coat")
        return scene_graph

    def get_camera_preset(self) -> CameraSpec:
        return CameraSpec(
            shot_type="medium chest-level portrait shot",
            camera_angle="eye-level compassionate angle",
            lens="85mm f/1.4 portrait lens",
            perspective="intimate direct perspective"
        )

    def get_lighting_preset(self) -> LightingSpec:
        return LightingSpec(
            primary_light="soft diffuse clinical daylight from frosted glass windows",
            color_temperature="6000K clean daylight",
            shadows="gentle soft fill shadows",
            mood="calm, empathetic, professional, reassuring"
        )

    def get_style_preset(self) -> StyleSpec:
        return StyleSpec(
            rendering_style="clean commercial medical photography",
            medium="high-end editorial photography",
            materials_and_textures=["sterile glass", "brushed stainless steel", "soft cotton scrubs"],
            color_palette=["hospital blue", "teal", "pure white", "warm skin tones"]
        )

    def get_domain_negative_prompts(self) -> List[str]:
        return ["dirty environment", "blood splatter", "horror aesthetic", "grainy texture"]


class CorporateMarketingDomain(BaseDomainPack):
    @property
    def domain_name(self) -> str:
        return "corporate"

    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        for node in scene_graph.nodes:
            if node.node_type == "subject":
                node.attributes.setdefault("clothing", "smart casual business attire")
        return scene_graph

    def get_camera_preset(self) -> CameraSpec:
        return CameraSpec(
            shot_type="medium wide environmental shot",
            camera_angle="eye-level dynamic angle",
            lens="35mm f/2.0 wide angle prime",
            perspective="modern corporate architectural view"
        )

    def get_lighting_preset(self) -> LightingSpec:
        return LightingSpec(
            primary_light="bright natural light from floor-to-ceiling office windows",
            color_temperature="5000K neutral daylight",
            shadows="soft ambient shadows",
            mood="optimistic, forward-thinking, sleek"
        )

    def get_style_preset(self) -> StyleSpec:
        return StyleSpec(
            rendering_style="premium brand lifestyle photography",
            medium="commercial advertising digital shot",
            materials_and_textures=["transparent glass walls", "natural oak wood", "matte aluminum"],
            color_palette=["navy blue", "slate grey", "warm wood tones", "accent white"]
        )

    def get_domain_negative_prompts(self) -> List[str]:
        return ["cluttered background", "poor lighting", "unprofessional clothing"]


class GeneralDomain(BaseDomainPack):
    @property
    def domain_name(self) -> str:
        return "general"

    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        return scene_graph

    def get_camera_preset(self) -> CameraSpec:
        return CameraSpec(
            shot_type="medium shot",
            camera_angle="eye-level",
            lens="50mm prime",
            perspective="balanced perspective"
        )

    def get_lighting_preset(self) -> LightingSpec:
        return LightingSpec(
            primary_light="balanced natural lighting",
            color_temperature="5500K daylight",
            shadows="soft natural shadows",
            mood="clear, realistic, informative"
        )

    def get_style_preset(self) -> StyleSpec:
        return StyleSpec(
            rendering_style="commercial photography",
            medium="digital 35mm photo",
            materials_and_textures=["authentic surface textures"],
            color_palette=["natural realistic color tones"]
        )

    def get_domain_negative_prompts(self) -> List[str]:
        return ["blurry", "low quality", "distorted faces", "bad anatomy"]


class EducationDomain(BaseDomainPack):
    @property
    def domain_name(self) -> str:
        return "education"

    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        has_subject = False
        for node in scene_graph.nodes:
            if node.node_type == "subject":
                has_subject = True
                node.attributes.setdefault("attire", "smart casual professional educator attire")
            elif node.node_type == "object":
                if "desk" in node.name.lower() or "board" in node.name.lower():
                    node.attributes.setdefault("setting", "bright modern classroom learning environment")

        if not has_subject:
            scene_graph.nodes.insert(0, SceneNode(
                id="educator_1",
                name="Teacher / Student Ambassador",
                node_type="subject",
                spatial_zone="foreground center-right",
                attributes={
                    "clothing": "smart casual educator attire, holding an informative clipboard or tablet",
                    "action": "leading an engaging awareness discussion with students in a bright classroom setting",
                    "emotion": "approachable, attentive, encouraging"
                }
            ))
        return scene_graph

    def get_camera_preset(self) -> CameraSpec:
        return CameraSpec(
            shot_type="medium eye-level shot",
            camera_angle="eye level",
            lens="50mm prime lens",
            perspective="three-quarter perspective"
        )

    def get_lighting_preset(self) -> LightingSpec:
        return LightingSpec(
            primary_light="bright natural daylight streaming through large classroom windows",
            color_temperature="4500K warm neutral daylight",
            shadows="soft ambient daylight shadows",
            mood="bright, welcoming, educational"
        )

    def get_style_preset(self) -> StyleSpec:
        return StyleSpec(
            rendering_style="clean contemporary photography",
            medium="35mm professional photography",
            materials_and_textures=["wooden desks", "whiteboard", "educational posters"],
            color_palette=["warm oak", "sky blue", "soft teal", "crisp white"]
        )

    def get_domain_negative_prompts(self) -> List[str]:
        return ["dark room", "industrial factory", "blurry", "text errors", "garbled letters", "gym workout", "athletic runner"]


# Register built-in domain packs
PluginRegistry.register_domain(WarehouseSafetyDomain())
PluginRegistry.register_domain(HealthcareDomain())
PluginRegistry.register_domain(CorporateMarketingDomain())
PluginRegistry.register_domain(EducationDomain())
PluginRegistry.register_domain(GeneralDomain())
