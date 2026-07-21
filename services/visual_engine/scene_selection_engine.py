"""
Scene Selection Engine module for selecting the single most visually dramatic and informative
climax frame from a multi-event narrative sequence.
"""

import logging
from typing import List, Optional
from services.visual_engine.schemas import ContentAnalysisResult, StoryEvent

logger = logging.getLogger(__name__)


class SceneSelectionEngine:
    """Evaluates narrative events to select the single frame that communicates the story most effectively."""

    def select_best_climax_frame(self, content_analysis: ContentAnalysisResult) -> StoryEvent:
        events = content_analysis.story_events
        if not events:
            # Fallback synthetic climax event
            return StoryEvent(
                step_number=1,
                event_description="Main subject in action during core visual moment",
                involved_characters=[c.name for c in content_analysis.main_characters],
                is_climax=True,
                visual_impact_score=1.0
            )

        # 1. Look for explicitly flagged climax events
        climax_candidates = [e for e in events if e.is_climax]
        if climax_candidates:
            selected = max(climax_candidates, key=lambda e: e.visual_impact_score)
            logger.info(f"Selected climax event via explicit tag: Step {selected.step_number} - '{selected.event_description}'")
            return selected

        # 2. Evaluate drama/climax keywords
        drama_keywords = ["collapse", "fall", "emergency", "rescue", "rush", "spill", "accident", "strike", "break", "breakthrough"]
        for e in events:
            if any(kw in e.event_description.lower() for kw in drama_keywords):
                e.is_climax = True
                e.visual_impact_score = 0.95
                logger.info(f"Selected climax event via keyword scoring: Step {e.step_number} - '{e.event_description}'")
                return e

        # 3. Fallback to event with highest visual impact score
        selected = max(events, key=lambda e: e.visual_impact_score)
        selected.is_climax = True
        logger.info(f"Selected climax event via max impact score: Step {selected.step_number} - '{selected.event_description}'")
        return selected
