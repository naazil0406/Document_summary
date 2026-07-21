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
        
        # Character extraction heuristics
        characters = []
        if "alex" in text.lower():
            characters.append(CharacterEntity(
                name="Alex",
                role="main",
                description="Male warehouse worker in mid-30s wearing safety helmet, high-vis orange vest, and durable work uniform",
                action="Rushing forward with urgent outstretched arms to support collapsing coworker",
                emotion="Alert, determined, deeply concerned",
                body_language="Leaning forward dynamically in quick mid-stride response"
            ))
        
        coworker_name = "Coworker" if "coworker" in text.lower() else "Worker"
        if "collapse" in text.lower() or "slow" in text.lower() or "tired" in text.lower() or "staggering" in text.lower():
            characters.append(CharacterEntity(
                name=coworker_name,
                role="supporting",
                description="Male logistics worker in high-vis vest and work shirt",
                action="Stumbling and knees buckling under heat strain while holding heavy load",
                emotion="Exhausted, flushed, distressed",
                body_language="Unsteady posture, drooping shoulders, heavy breathing stance"
            ))

        if not characters:
            characters.append(CharacterEntity(
                name="Primary Subject",
                role="main",
                description="Professional worker in appropriate domain attire",
                action="Performing core task",
                emotion="Focused and professional",
                body_language="Engaged body position"
            ))

        # Objects & Human Performance Tools extraction
        objects = []
        if any(w in text.lower() for w in ["pallet", "load", "box", "rack"]):
            objects.append(ObjectEntity(name="Pallet Jack with Heavy Cargo", category="equipment", importance="high", visual_details="Industrial yellow hydraulic pallet truck holding stacked wooden boxes"))
            objects.append(ObjectEntity(name="Warehouse Storage Racks", category="building", importance="medium", visual_details="Multi-tier heavy steel shelving loaded with boxed inventory"))
        if any(w in text.lower() for w in ["helmet", "vest"]):
            objects.append(ObjectEntity(name="Safety Helmet", category="equipment", importance="high", visual_details="Bright yellow OSHA approved hard hat"))

        # Extract Human Performance Tools mentioned in text
        hpt_tools = []
        if "rate your state" in text.lower() or "rys" in text.lower():
            hpt_tools.append("Rate Your State (RYS) Checklist")
        if "anticipating error" in text.lower():
            hpt_tools.append("Anticipating Error Pre-task Planning Board")
        if "close calls" in text.lower():
            hpt_tools.append("Close Calls Incident Log")
        if "habit reminder" in text.lower():
            hpt_tools.append("Habit Reminder Visual Indicator")
        if "rys supervisor conversation" in text.lower():
            hpt_tools.append("RYS Supervisor Conversation Form")

        for tool in hpt_tools:
            objects.append(ObjectEntity(name=tool, category="tool", importance="medium", visual_details=f"Enterprise Human Performance Safety Tool: {tool}"))

        # Domain-aware environment heuristics
        weather = "hot midday summer heat" if any(w in text.lower() for w in ["sun", "heat", "hot", "summer"]) else "clear indoor environment"
        time_of_day = "midday" if "midday" in text.lower() or "afternoon" in text.lower() else "daytime"

        if domain == "warehouse":
            loc, btype = "Industrial warehouse loading bay floor", "logistics facility"
        elif domain == "healthcare":
            loc, btype = "Clinical consultation room", "medical facility"
        elif domain == "education":
            loc, btype = "Bright modern classroom learning environment", "educational campus"
        elif domain == "corporate":
            loc, btype = "Executive corporate conference room", "corporate office"
        else:
            loc, btype = "Modern professional indoor environment", "commercial facility"

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
