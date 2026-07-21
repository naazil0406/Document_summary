"""
Data models and JSON schemas for the Universal AI Visual Content Engine.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class UserIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_request: str = Field(..., description="Original user topic or query")
    content_type: str = Field(default="Scenario", description="Content type requested (e.g. Recall Card, Infographic, Scenario)")
    domain: str = Field(default="general", description="Inferred or explicitly specified domain (e.g., warehouse, healthcare, marketing)")
    communication_purpose: str = Field(default="education", description="Primary goal: safety, marketing, education, corporate, training, etc.")
    target_audience: str = Field(default="general professional", description="Target audience demographic or professional tier")
    aspect_ratio: str = Field(default="16:9", description="Target aspect ratio for visual output (16:9, 1:1, 9:16, 4:3)")
    style_preference: Optional[str] = Field(default="commercial_photography", description="Requested or default visual rendering style")


class ContentGenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_content: str = Field(..., description="Generated text content acting as single source of truth")
    title: str = Field(default="", description="Headline or title for content")
    core_message: str = Field(default="", description="Central thesis or safety/business message")
    key_takeaways: List[str] = Field(default_factory=list, description="Bullet points summarizing core points")


class CharacterEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Character name or identity (e.g., Alex, Coworker)")
    role: str = Field(default="main", description="Role: main, supporting, background")
    description: str = Field(..., description="Physical appearance, age, attire, safety gear")
    action: str = Field(..., description="Current physical movement or action in scene")
    emotion: str = Field(..., description="Facial expression and emotional state")
    body_language: str = Field(default="", description="Posture and bodily positioning")


class ObjectEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Object name (e.g., pallet jack, heavy pallet, safety helmet)")
    category: str = Field(default="equipment", description="Category: tool, vehicle, equipment, building, prop")
    importance: str = Field(default="high", description="Visual priority: high, medium, low")
    visual_details: str = Field(default="", description="Color, material, condition, position")


class EnvironmentEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: str = Field(..., description="Primary location (e.g., industrial warehouse floor)")
    building_type: str = Field(default="warehouse", description="Structure type")
    weather: str = Field(default="hot summer day", description="Weather condition")
    season: str = Field(default="summer", description="Season")
    time_of_day: str = Field(default="midday", description="Time of day")
    lighting_clues: str = Field(default="harsh bright sunlight, high contrast ambient shadows", description="Lighting environment")


class StoryEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_number: int = Field(..., description="Sequential position in narrative")
    event_description: str = Field(..., description="Summary of action in this frame")
    involved_characters: List[str] = Field(default_factory=list, description="Names of involved characters")
    is_climax: bool = Field(default=False, description="Whether this event represents the narrative climax")
    visual_impact_score: float = Field(default=0.5, description="Visual drama and message clarity score (0.0 to 1.0)")


class MessageContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message_type: str = Field(default="safety", description="Type: safety, marketing, educational, corporate, brand")
    core_takeaway: str = Field(..., description="Underlying business or safety directive")


class ContentAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    main_characters: List[CharacterEntity] = Field(default_factory=list)
    supporting_characters: List[CharacterEntity] = Field(default_factory=list)
    objects: List[ObjectEntity] = Field(default_factory=list)
    environment: EnvironmentEntity = Field(default_factory=lambda: EnvironmentEntity(location="indoor setting"))
    story_events: List[StoryEvent] = Field(default_factory=list)
    selected_climax_event: Optional[StoryEvent] = Field(default=None)
    message_context: MessageContext = Field(default_factory=lambda: MessageContext(core_takeaway="Default message"))
    overall_mood: str = Field(default="urgent, realistic, professional")


class SceneNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(..., description="Unique node ID")
    name: str = Field(..., description="Name of subject/object")
    node_type: str = Field(..., description="subject, object, environment, lighting, text_copy")
    spatial_zone: str = Field(default="midground", description="foreground, midground, background")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Visual attributes like posture, attire, material, color")


class SceneEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: str = Field(...)
    target_id: str = Field(...)
    relation: str = Field(..., description="Spatial or behavioral relationship (e.g. rushing_towards, collapsing_next_to)")


class SceneGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: List[SceneNode] = Field(default_factory=list)
    edges: List[SceneEdge] = Field(default_factory=list)
    focal_node_id: str = Field(default="")
    climax_moment_summary: str = Field(default="")
    spatial_layout: Dict[str, List[str]] = Field(default_factory=lambda: {"foreground": [], "midground": [], "background": []})
    environment_summary: str = Field(default="")
    mood_and_atmosphere: str = Field(default="")
    domain_type: str = Field(default="warehouse")


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shot_type: str = Field(default="medium wide shot")
    camera_angle: str = Field(default="slightly low angle")
    lens: str = Field(default="50mm f/1.8 prime lens")
    perspective: str = Field(default="dynamic three-quarter view")


class LightingSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_light: str = Field(default="intense direct sunlight streaming from high windows")
    color_temperature: str = Field(default="warm 5500K golden midday light")
    shadows: str = Field(default="crisp defined shadows with soft fill")
    mood: str = Field(default="dramatic, high visual urgency")


class StyleSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rendering_style: str = Field(default="commercial documentary photography")
    medium: str = Field(default="high-resolution 35mm digital photography")
    materials_and_textures: List[str] = Field(default_factory=lambda: ["high-visibility vest fabric", "dusty concrete", "worn steel racks"])
    color_palette: List[str] = Field(default_factory=lambda: ["safety orange", "industrial yellow", "warm neutral tones"])


class LayoutSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    copy_space: str = Field(default="clean negative space in top left third for text placement")
    visual_hierarchy: str = Field(default="primary focus on collapsing worker and rushing colleague, secondary focus on warehouse racks")
    aspect_ratio: str = Field(default="16:9")


class CompiledPromptSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    positive_prompt: str = Field(..., description="The fully compiled detailed image prompt derived strictly from Scene Graph")
    negative_prompt: str = Field(default="", description="Negative prompt specifying elements to avoid")
    camera: CameraSpec = Field(default_factory=CameraSpec)
    lighting: LightingSpec = Field(default_factory=LightingSpec)
    style: StyleSpec = Field(default_factory=StyleSpec)
    layout: LayoutSpec = Field(default_factory=LayoutSpec)
    applied_rules: List[str] = Field(default_factory=list, description="Rules applied during prompt compilation")


class ConsistencyReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_consistent: bool = Field(..., description="True if prompt meets consistency threshold")
    overall_score: float = Field(..., description="Overall cross-modal consistency score 0.0 - 1.0")
    character_score: float = Field(default=1.0)
    object_score: float = Field(default=1.0)
    action_score: float = Field(default=1.0)
    environment_score: float = Field(default=1.0)
    message_score: float = Field(default=1.0)
    discrepancies: List[str] = Field(default_factory=list)
    refinement_feedback: str = Field(default="")


class VisualEngineOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: UserIntent
    generated_content: ContentGenerationOutput
    content_analysis: ContentAnalysisResult
    scene_graph: SceneGraph
    prompt_spec: CompiledPromptSpec
    consistency_report: ConsistencyReport
    image_bytes_base64: Optional[str] = Field(default=None, description="Generated image in base64 if image model was called")
    image_provider_used: Optional[str] = Field(default=None)
