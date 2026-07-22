"""
Intent Analyzer module for determining user intent, domain, goal, and visual preferences.
"""

import json
import logging
from typing import Optional
from services.visual_engine.schemas import UserIntent

logger = logging.getLogger(__name__)


class IntentAnalyzer:
    """Analyzes user requests to infer domain, goal, target audience, and layout constraints."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def analyze_intent(
        self,
        user_request: str,
        domain_override: Optional[str] = None,
        aspect_ratio: str = "16:9",
        style_preference: Optional[str] = None,
    ) -> UserIntent:
        """Parse raw request using heuristic rules or LLM fallback into UserIntent."""
        request_lower = user_request.lower()

        # Content Type detection
        content_type = "Scenario"
        raw_topic = user_request.strip()
        known_content_types = [
            "Recall Card", "AI Image", "Infographic", "Flashcard", "Scenario",
            "Spot the Mistake Challenge", "Daily Quiz", "Fun Fact",
            "Reflection Question", "Safety / Best Practice Tip", "Daily Tip"
        ]

        if ":" in user_request:
            prefix, rest = user_request.split(":", 1)
            prefix_stripped = prefix.strip()
            for ctype in known_content_types:
                if prefix_stripped.lower() == ctype.lower():
                    content_type = ctype
                    raw_topic = rest.strip()
                    break

        if content_type == "Scenario" and ":" not in user_request:
            for ctype in known_content_types:
                if ctype.lower() in request_lower:
                    content_type = ctype
                    break

        # Domain heuristic detection
        domain = domain_override
        if not domain:
            if any(w in request_lower for w in ["warehouse", "forklift", "pallet", "heat stress", "factory", "worker", "safety vest", "hardhat"]):
                domain = "warehouse"
            elif any(w in request_lower for w in ["doctor", "patient", "hospital", "clinic", "medical", "nurse", "healthcare"]):
                domain = "healthcare"
            elif any(w in request_lower for w in ["school", "student", "teacher", "classroom", "education", "campus", "academic"]):
                domain = "education"
            elif any(w in request_lower for w in ["office", "executive", "meeting", "strategy", "brand", "marketing", "corporate"]):
                domain = "corporate"
            elif any(w in request_lower for w in ["code", "software", "ai", "cloud", "server", "data center", "tech"]):
                domain = "technology"
            else:
                domain = "general"

        # Communication purpose heuristic
        purpose = "education"
        if any(w in request_lower for w in ["safety", "hazard", "emergency", "caution", "injury", "protect", "heat"]):
            purpose = "safety"
        elif any(w in request_lower for w in ["sell", "product", "launch", "campaign", "ad", "marketing", "banner"]):
            purpose = "marketing"
        elif any(w in request_lower for w in ["train", "how-to", "procedure", "guide", "sop"]):
            purpose = "training"

        # Style preference fallback
        style = style_preference or "commercial_photography"
        if content_type == "Infographic" or "infographic" in request_lower or "diagram" in request_lower:
            style = "infographic_illustration"
        elif "cinematic" in request_lower or "movie" in request_lower or "story" in request_lower:
            style = "cinematic_storytelling"

        return UserIntent(
            raw_request=raw_topic,
            content_type=content_type,
            domain=domain,
            communication_purpose=purpose,
            target_audience="frontline workforce and safety managers" if domain == "warehouse" else "general professional",
            aspect_ratio=aspect_ratio,
            style_preference=style
        )