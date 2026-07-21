"""
Content & Image Consistency Validator module.
Compares Generated Content vs Generated Image Prompt across 10 key dimensions
and triggers automatic refinement if inconsistencies exist.
"""

import logging
from typing import List, Tuple
from services.visual_engine.schemas import (
    ContentAnalysisResult,
    CompiledPromptSpec,
    ConsistencyReport,
)

logger = logging.getLogger(__name__)


class ConsistencyValidator:
    """Audits cross-modal consistency between Generated Content and Compiled Image Prompt."""

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold

    def validate(
        self,
        content_analysis: ContentAnalysisResult,
        prompt_spec: CompiledPromptSpec,
    ) -> ConsistencyReport:
        prompt_text = prompt_spec.positive_prompt.lower()
        discrepancies: List[str] = []

        # 1. Character Check
        char_matches = 0
        all_chars = content_analysis.main_characters + content_analysis.supporting_characters
        for char in all_chars:
            if char.name.lower() in prompt_text or any(w in prompt_text for w in char.name.lower().split()):
                char_matches += 1
            else:
                discrepancies.append(f"Character '{char.name}' missing or unreferenced in visual prompt.")

        char_score = char_matches / max(len(all_chars), 1)

        # 2. Action & Climax Check
        climax_desc = content_analysis.selected_climax_event.event_description if content_analysis.selected_climax_event else ""
        action_words = [w for w in climax_desc.lower().split() if len(w) > 4]
        action_matches = sum(1 for w in action_words if w in prompt_text)
        action_score = action_matches / max(len(action_words), 1) if action_words else 1.0

        if action_score < 0.3:
            discrepancies.append("Primary climax action is inadequately reflected in prompt composition.")

        # 3. Environment & Weather Check
        env = content_analysis.environment
        env_score = 1.0
        if env.weather.lower() not in prompt_text and not any(w in prompt_text for w in ["sun", "daylight", "heat", "weather"]):
            env_score -= 0.3
            discrepancies.append(f"Weather condition '{env.weather}' missing from prompt background details.")

        # 4. Object Check
        obj_matches = sum(1 for obj in content_analysis.objects if obj.name.lower() in prompt_text or any(w in prompt_text for w in obj.name.lower().split()))
        obj_score = obj_matches / max(len(content_analysis.objects), 1) if content_analysis.objects else 1.0

        # 5. Core Message & Mood Check
        msg_takeaway = content_analysis.message_context.core_takeaway.lower()
        msg_keywords = [w for w in msg_takeaway.split() if len(w) > 4]
        msg_matches = sum(1 for w in msg_keywords if w in prompt_text or w in prompt_spec.negative_prompt.lower())
        message_score = msg_matches / max(len(msg_keywords), 1) if msg_keywords else 1.0

        # Overall Score Calculation
        overall_score = (
            (char_score * 0.3) +
            (action_score * 0.3) +
            (env_score * 0.15) +
            (obj_score * 0.15) +
            (message_score * 0.10)
        )

        is_consistent = overall_score >= self.threshold and len(discrepancies) == 0

        refinement_feedback = ""
        if not is_consistent:
            refinement_feedback = "Auto-refinement needed: " + "; ".join(discrepancies)

        return ConsistencyReport(
            is_consistent=is_consistent,
            overall_score=round(overall_score, 3),
            character_score=round(char_score, 3),
            object_score=round(obj_score, 3),
            action_score=round(action_score, 3),
            environment_score=round(env_score, 3),
            message_score=round(message_score, 3),
            discrepancies=discrepancies,
            refinement_feedback=refinement_feedback
        )
