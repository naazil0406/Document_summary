"""
Plugin architecture and registry for Domain Packs and Visual Template Packs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from services.visual_engine.schemas import SceneGraph, CameraSpec, LightingSpec, StyleSpec, LayoutSpec


class BaseDomainPack(ABC):
    """Abstract Base Class for Domain Packs."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        pass

    @abstractmethod
    def enhance_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        """Inject domain-specific visual details or constraints into Scene Graph."""
        pass

    @abstractmethod
    def get_camera_preset(self) -> CameraSpec:
        pass

    @abstractmethod
    def get_lighting_preset(self) -> LightingSpec:
        pass

    @abstractmethod
    def get_style_preset(self) -> StyleSpec:
        pass

    @abstractmethod
    def get_domain_negative_prompts(self) -> List[str]:
        pass


class BaseTemplatePack(ABC):
    """Abstract Base Class for Template Packs (Visual Composition & Formatting)."""

    @property
    @abstractmethod
    def template_name(self) -> str:
        pass

    @abstractmethod
    def apply_layout(self, scene_graph: SceneGraph, aspect_ratio: str = "16:9") -> LayoutSpec:
        pass

    @abstractmethod
    def format_prompt(
        self,
        scene_graph: SceneGraph,
        camera: CameraSpec,
        lighting: LightingSpec,
        style: StyleSpec,
        layout: LayoutSpec
    ) -> str:
        pass


class PluginRegistry:
    """Central registry for dynamic domain and template pack plugins."""

    _domains: Dict[str, BaseDomainPack] = {}
    _templates: Dict[str, BaseTemplatePack] = {}

    @classmethod
    def register_domain(cls, domain_pack: BaseDomainPack) -> None:
        cls._domains[domain_pack.domain_name.lower()] = domain_pack

    @classmethod
    def register_template(cls, template_pack: BaseTemplatePack) -> None:
        cls._templates[template_pack.template_name.lower()] = template_pack

    @classmethod
    def get_domain(cls, domain_name: str) -> Optional[BaseDomainPack]:
        name = domain_name.lower()
        if name in cls._domains:
            return cls._domains[name]
        # Fallback to general if available
        return cls._domains.get("general") or next(iter(cls._domains.values()), None)

    @classmethod
    def get_template(cls, template_name: str) -> Optional[BaseTemplatePack]:
        name = template_name.lower()
        if name in cls._templates:
            return cls._templates[name]
        return cls._templates.get("commercial_photography") or next(iter(cls._templates.values()), None)

    @classmethod
    def list_domains(cls) -> List[str]:
        return list(cls._domains.keys())

    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls._templates.keys())
