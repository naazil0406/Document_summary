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

        # Domain heuristic detection
        domain = domain_override
        if not domain:
            if any(w in request_lower for w in ["warehouse", "forklift", "pallet", "heat stress", "factory", "worker", "safety vest", "hardhat"]):
                domain = "warehouse"
            elif any(w in request_lower for w in ["doctor", "patient", "hospital", "clinic", "medical", "nurse", "healthcare"]):
                domain = "healthcare"
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
        if "infographic" in request_lower or "diagram" in request_lower:
            style = "infographic_illustration"
        elif "cinematic" in request_lower or "movie" in request_lower or "story" in request_lower:
            style = "cinematic_storytelling"

        return UserIntent(
            raw_request=user_request,
            domain=domain,
            communication_purpose=purpose,
            target_audience="frontline workforce and safety managers" if domain == "warehouse" else "general professional",
            aspect_ratio=aspect_ratio,
            style_preference=style
        )
