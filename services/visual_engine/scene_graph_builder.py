"""
Scene Graph Builder module for constructing pure mathematical/structural Scene Graphs
from ContentAnalysisResult without generating text prompt strings.
"""

import logging
from typing import List, Dict, Any

from services.visual_engine.schemas import (
    ContentAnalysisResult,
    SceneGraph,
    SceneNode,
    SceneEdge,
    CharacterEntity,
    ObjectEntity,
    EnvironmentEntity,
)

logger = logging.getLogger(__name__)


class SceneGraphBuilder:
    """Converts structured narrative extraction into a pure spatial and relational SceneGraph."""

    def build_scene_graph(self, content_analysis: ContentAnalysisResult, domain: str = "warehouse") -> SceneGraph:
        nodes: List[SceneNode] = []
        edges: List[SceneEdge] = []
        spatial_layout: Dict[str, List[str]] = {"foreground": [], "midground": [], "background": []}

        # 1. Subject Nodes (Main Characters -> Midground / Foreground focal)
        focal_node_id = ""
        for i, char in enumerate(content_analysis.main_characters, 1):
            node_id = f"node_char_main_{i}"
            if not focal_node_id:
                focal_node_id = node_id

            zone = "foreground" if i == 1 else "midground"
            spatial_layout[zone].append(node_id)

            nodes.append(SceneNode(
                node_id=node_id,
                name=f"{char.name} ({char.description})",
                node_type="subject",
                spatial_zone=zone,
                attributes={
                    "role": char.role,
                    "action": char.action,
                    "emotion": char.emotion,
                    "body_language": char.body_language,
                    "clothing": char.description
                }
            ))

        # 2. Supporting Characters -> Midground
        for i, char in enumerate(content_analysis.supporting_characters, 1):
            node_id = f"node_char_supp_{i}"
            spatial_layout["midground"].append(node_id)

            nodes.append(SceneNode(
                node_id=node_id,
                name=f"{char.name} ({char.description})",
                node_type="subject",
                spatial_zone="midground",
                attributes={
                    "role": char.role,
                    "action": char.action,
                    "emotion": char.emotion,
                    "body_language": char.body_language,
                    "clothing": char.description
                }
            ))

            # Add relationship edge between main character and supporting character
            if focal_node_id:
                edges.append(SceneEdge(
                    source_id=focal_node_id,
                    target_id=node_id,
                    relation="rushing_towards_and_reaching_to_assist"
                ))

        # 3. Object Nodes -> Midground / Foreground
        for i, obj in enumerate(content_analysis.objects, 1):
            node_id = f"node_obj_{i}"
            zone = "foreground" if obj.importance == "high" and i == 1 else "midground"
            spatial_layout[zone].append(node_id)

            nodes.append(SceneNode(
                node_id=node_id,
                name=f"{obj.name}",
                node_type="object",
                spatial_zone=zone,
                attributes={
                    "category": obj.category,
                    "importance": obj.importance,
                    "visual_details": obj.visual_details
                }
            ))

            # Add spatial relation edge to supporting character or main character
            if nodes and len(nodes) > 1:
                target = nodes[1].node_id
                edges.append(SceneEdge(
                    source_id=target,
                    target_id=node_id,
                    relation="collapsing_adjacent_to" if "pallet" in obj.name.lower() else "positioned_near"
                ))

        # 4. Environment & Lighting Nodes -> Background
        env = content_analysis.environment
        env_node_id = "node_env_1"
        spatial_layout["background"].append(env_node_id)
        nodes.append(SceneNode(
            node_id=env_node_id,
            name=f"{env.location} ({env.building_type})",
            node_type="environment",
            spatial_zone="background",
            attributes={
                "weather": env.weather,
                "season": env.season,
                "time_of_day": env.time_of_day,
                "lighting_clues": env.lighting_clues
            }
        ))

        # Climax summary
        climax_summary = (
            content_analysis.selected_climax_event.event_description
            if content_analysis.selected_climax_event
            else "Primary key visual moment of content"
        )

        return SceneGraph(
            nodes=nodes,
            edges=edges,
            focal_node_id=focal_node_id or (nodes[0].node_id if nodes else ""),
            climax_moment_summary=climax_summary,
            spatial_layout=spatial_layout,
            environment_summary=f"{env.location}, {env.weather}, {env.time_of_day}",
            mood_and_atmosphere=content_analysis.overall_mood,
            domain_type=domain
        )
