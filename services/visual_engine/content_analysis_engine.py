"""
Content Analysis Engine module for extracting structured entities, visual clues,
emotions, actions, and narrative climax from generated text content.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from services.visual_engine.schemas import (
    ContentAnalysisResult,
    CharacterEntity,
    ObjectEntity,
    EnvironmentEntity,
    StoryEvent,
    MessageContext,
)

logger = logging.getLogger(__name__)


class ContentAnalysisEngine:
    """Deeply parses generated text content to produce structured ContentAnalysisResult JSON."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def analyze_content(self, generated_content: str, domain: str = "general") -> ContentAnalysisResult:
        """Parse content into structured JSON using LLM or rule-based fallback."""
        if self.llm_service:
            try:
                return self._analyze_with_llm(generated_content, domain)
            except Exception as e:
                logger.warning(f"LLM content analysis failed ({e}), falling back to structured extractor.")

        return self._analyze_with_rules(generated_content, domain)

    def _analyze_with_llm(self, text: str, domain: str) -> ContentAnalysisResult:
        prompt = f"""You are a Computer Vision Director and Storyboard Architect.
Analyze the following text content and extract all visual, narrative, spatial, and emotional details into structured JSON.

Generated Content:
\"\"\"{text}\"\"\"

Target Domain: {domain}

Output ONLY valid JSON matching this exact structure:
{{
  "main_characters": [
    {{"name": "...", "role": "main", "description": "...", "action": "...", "emotion": "...", "body_language": "..."}}
  ],
  "supporting_characters": [
    {{"name": "...", "role": "supporting", "description": "...", "action": "...", "emotion": "...", "body_language": "..."}}
  ],
  "objects": [
    {{"name": "...", "category": "equipment/tool/vehicle/prop", "importance": "high/medium/low", "visual_details": "..."}}
  ],
  "environment": {{
    "location": "...", "building_type": "...", "weather": "...", "season": "...", "time_of_day": "...", "lighting_clues": "..."
  }},
  "story_events": [
    {{"step_number": 1, "event_description": "...", "involved_characters": ["..."], "is_climax": false, "visual_impact_score": 0.8}}
  ],
  "message_context": {{
    "message_type": "safety/marketing/educational/brand", "core_takeaway": "..."
  }},
  "overall_mood": "urgent, realistic, professional"
}}"""

        response_text = self.llm_service._call_llm(
            system_prompt="Extract structured vision schema from text. Respond ONLY with valid JSON.",
            user_prompt=prompt,
        )
        # Parse JSON from response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return ContentAnalysisResult(**data)
        
        return self._analyze_with_rules(text, domain)

    def _analyze_with_rules(self, text: str, domain: str) -> ContentAnalysisResult:
        """Rule-based NLP heuristic parser for offline/fallback execution."""
        # Split text into sentences for fine-grained multi-event story progression timeline
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 3]
        if not sentences:
            sentences = [text.strip()]
        
        text_lower = text.lower()
        characters = []
        objects = []

        # Text-driven scene heuristic detection
        is_driving = any(w in text_lower for w in ["driving", "car", "traffic", "road", "merge", "vehicle", "commute", "steering", "following distance", "school zone", "school drop-off", "drop-offs", "engine", "behind the wheel"])
        is_warehouse = any(w in text_lower for w in ["pallet", "forklift", "loading bay", "cargo", "warehouse", "heat stress", "staggering", "collapsing"])
        is_classroom = any(w in text_lower for w in ["classroom", "teacher", "blackboard", "whiteboard", "student", "desk", "lesson plan"])
        is_medical = any(w in text_lower for w in ["hospital", "patient", "clinic", "doctor", "nurse", "medical"])

        if is_driving:
            characters.append(CharacterEntity(
                name="Commuter Driver",
                role="main",
                description="Adult commuter driver sitting at the steering wheel inside a car",
                action="Glancing at wristwatch while navigating heavy morning commute traffic near school zone",
                emotion="Alert, focused, mindful of traffic ahead",
                body_language="Hands on steering wheel, leaning forward attentively"
            ))
            objects.append(ObjectEntity(
                name="Car Dashboard and Steering Wheel",
                category="vehicle",
                importance="high",
                visual_details="Modern automobile interior view from driver perspective showing steering wheel and windshield"
            ))
            objects.append(ObjectEntity(
                name="Morning Commute Traffic",
                category="vehicle",
                importance="high",
                visual_details="Commuter vehicles merged on suburban road near school zone lights"
            ))
            loc, btype = "Suburban roadway in morning commute traffic near school zone", "vehicle interior and roadway"
            weather, time_of_day = "bright morning sunlight", "morning"
        elif is_warehouse or "alex" in text_lower:
            if "alex" in text_lower:
                characters.append(CharacterEntity(
                    name="Alex",
                    role="main",
                    description="Male warehouse worker in high-vis orange vest and safety helmet",
                    action="Rushing forward to support collapsing coworker",
                    emotion="Alert, determined, deeply concerned",
                    body_language="Leaning forward dynamically in quick mid-stride response"
                ))
            if "collapse" in text_lower or "staggering" in text_lower or "coworker" in text_lower:
                characters.append(CharacterEntity(
                    name="Coworker",
                    role="supporting",
                    description="Logistics worker in high-vis vest and work shirt",
                    action="Stumbling under heat strain while holding heavy load",
                    emotion="Exhausted, flushed, distressed",
                    body_language="Unsteady posture, drooping shoulders"
                ))
            objects.append(ObjectEntity(name="Pallet Jack with Heavy Cargo", category="equipment", importance="high", visual_details="Industrial yellow hydraulic pallet truck"))
            loc, btype = "Industrial warehouse loading bay floor", "logistics facility"
            weather, time_of_day = "hot midday summer heat", "midday"
        elif is_classroom:
            characters.append(CharacterEntity(
                name="Educator",
                role="main",
                description="Professional teacher standing in front of classroom",
                action="Engaging students in classroom discussion",
                emotion="Attentive, encouraging",
                body_language="Standing upright, gesturing towards whiteboard"
            ))
            objects.append(ObjectEntity(name="Classroom Desks and Whiteboard", category="building", importance="high", visual_details="Modern wooden desks and interactive learning board"))
            loc, btype = "Bright modern classroom learning environment", "educational campus"
            weather, time_of_day = "bright daylight", "daytime"
        elif is_medical:
            characters.append(CharacterEntity(
                name="Healthcare Professional",
                role="main",
                description="Physician or nurse wearing clean medical scrubs",
                action="Consulting with patient in clinic room",
                emotion="Empathetic, attentive",
                body_language="Seated attentively opposite patient"
            ))
            loc, btype = "Clinical consultation room", "medical facility"
            weather, time_of_day = "clear daylight", "daytime"
        else:
            characters.append(CharacterEntity(
                name="Primary Subject",
                role="main",
                description="Professional in appropriate domain attire",
                action="Performing core task",
                emotion="Focused and professional",
                body_language="Engaged posture"
            ))
            if domain == "warehouse":
                loc, btype = "Industrial warehouse floor", "logistics facility"
            elif domain == "healthcare":
                loc, btype = "Clinical consultation room", "medical facility"
            elif domain == "education":
                loc, btype = "Modern educational facility corridor", "educational campus"
            elif domain == "corporate":
                loc, btype = "Executive corporate conference room", "corporate office"
            else:
                loc, btype = "Modern professional indoor environment", "commercial facility"
            weather, time_of_day = "clear daylight", "daytime"

        # Extract Human Performance Tools mentioned in text
        hpt_tools = []
        if "rate your state" in text_lower or "rys" in text_lower:
            hpt_tools.append("Rate Your State (RYS) Checklist")
        if "anticipating error" in text_lower:
            hpt_tools.append("Anticipating Error Pre-task Planning Board")
        if "close calls" in text_lower:
            hpt_tools.append("Close Calls Incident Log")
        if "habit reminder" in text_lower:
            hpt_tools.append("Habit Reminder Visual Indicator")
        if "rys supervisor conversation" in text_lower:
            hpt_tools.append("RYS Supervisor Conversation Form")

        for tool in hpt_tools:
            objects.append(ObjectEntity(name=tool, category="tool", importance="medium", visual_details=f"Enterprise Human Performance Safety Tool: {tool}"))

        environment = EnvironmentEntity(
            location=loc,
            building_type=btype,
            weather=weather,
            season="summer",
            time_of_day=time_of_day,
            lighting_clues="Bright natural ambient lighting creating clear visibility"
        )

        # Story events timeline heuristics
        events = []
        for i, sentence in enumerate(sentences, 1):
            is_climax = any(kw in sentence.lower() for kw in ["collapsed", "staggering", "fell", "rushed", "emergency"])
            impact = 0.95 if is_climax else 0.4 + (i * 0.05)
            events.append(StoryEvent(
                step_number=i,
                event_description=sentence,
                involved_characters=[c.name for c in characters],
                is_climax=is_climax,
                visual_impact_score=min(impact, 1.0)
            ))

        # Identify climax event
        climax_event = next((e for e in events if e.is_climax), None)
        if not climax_event and events:
            climax_event = max(events, key=lambda e: e.visual_impact_score)

        main_chars = [c for c in characters if c.role == "main"]
        supp_chars = [c for c in characters if c.role != "main"]

        takeaway = "Recognize heat stress early and apply Human Performance Tools like Rate Your State (RYS) and Anticipating Error."
        if hpt_tools:
            takeaway += f" Tools utilized: {', '.join(hpt_tools)}."

        return ContentAnalysisResult(
            main_characters=main_chars,
            supporting_characters=supp_chars,
            objects=objects,
            environment=environment,
            story_events=events,
            selected_climax_event=climax_event,
            message_context=MessageContext(
                message_type="safety" if domain == "warehouse" else "general",
                core_takeaway=takeaway
            ),
            overall_mood="urgent, realistic, professional"
        )