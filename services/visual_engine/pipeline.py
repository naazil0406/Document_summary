"""
Full End-to-End Orchestrator Pipeline for the Universal AI Visual Content Engine.
"""

import base64
import logging
from typing import Optional, Dict, Any

from services.visual_engine.schemas import (
    UserIntent,
    ContentGenerationOutput,
    ContentAnalysisResult,
    SceneGraph,
    CompiledPromptSpec,
    ConsistencyReport,
    VisualEngineOutput,
)
from services.visual_engine.intent_analyzer import IntentAnalyzer
from services.visual_engine.content_analysis_engine import ContentAnalysisEngine
from services.visual_engine.scene_graph_builder import SceneGraphBuilder
from services.visual_engine.scene_selection_engine import SceneSelectionEngine
from services.visual_engine.prompt_compiler import UniversalPromptCompiler
from services.visual_engine.consistency_validator import ConsistencyValidator

logger = logging.getLogger(__name__)


class UniversalVisualContentEngine:
    """Enterprise Orchestrator for content-driven AI visual generation."""

    def __init__(
        self,
        llm_service=None,
        retriever_service=None,
        image_service=None,
        consistency_threshold: float = 0.80,
        max_refinement_attempts: int = 2
    ):
        self.llm_service = llm_service
        self.retriever_service = retriever_service
        self.image_service = image_service

        self.intent_analyzer = IntentAnalyzer(llm_service=llm_service)
        self.content_analyzer = ContentAnalysisEngine(llm_service=llm_service)
        self.scene_graph_builder = SceneGraphBuilder()
        self.scene_selector = SceneSelectionEngine()
        self.prompt_compiler = UniversalPromptCompiler()
        self.validator = ConsistencyValidator(threshold=consistency_threshold)
        self.max_refinement_attempts = max_refinement_attempts

    def generate_content_and_visual(
        self,
        user_request: str,
        domain_override: Optional[str] = None,
        style_override: Optional[str] = None,
        aspect_ratio: str = "16:9",
        generate_image_bytes: bool = True
    ) -> VisualEngineOutput:
        """Executes the complete 10-step visual content pipeline."""
        logger.info(f"Step 1: Analyzing user intent for request: '{user_request}'")
        intent: UserIntent = self.intent_analyzer.analyze_intent(
            user_request=user_request,
            domain_override=domain_override,
            aspect_ratio=aspect_ratio,
            style_preference=style_override
        )

        # Step 2: Knowledge Retrieval (optional)
        retrieved_context = ""
        if self.retriever_service:
            try:
                retrieved_chunks = self.retriever_service.retrieve(intent.raw_request)
                retrieved_context = "\n".join(chunk.text for chunk in retrieved_chunks)
            except Exception as e:
                logger.warning(f"Retrieval step failed ({e}), continuing without context.")

        # Step 3: Content Generator (Content is Single Source of Truth)
        logger.info("Step 3: Generating core text content (Single Source of Truth)")
        generated_content: ContentGenerationOutput = self._generate_text_content(
            intent=intent,
            retrieved_context=retrieved_context
        )

        # Step 4: Content Analysis Engine
        logger.info("Step 4: Running Content Analysis Engine to extract structured vision schema")
        content_analysis: ContentAnalysisResult = self.content_analyzer.analyze_content(
            generated_content=generated_content.raw_content,
            domain=intent.domain
        )

        # Step 5: Scene Selection Engine (Select key climax frame)
        logger.info("Step 5: Running Scene Selection Engine to identify climax frame")
        climax_event = self.scene_selector.select_best_climax_frame(content_analysis)
        content_analysis.selected_climax_event = climax_event

        # Step 6: Scene Graph Builder
        logger.info("Step 6: Building pure mathematical Scene Graph")
        scene_graph: SceneGraph = self.scene_graph_builder.build_scene_graph(
            content_analysis=content_analysis,
            domain=intent.domain
        )

        # Step 7: Universal Prompt Compiler
        logger.info("Step 7: Compiling image prompt strictly from Scene Graph")
        prompt_spec: CompiledPromptSpec = self.prompt_compiler.compile_prompt(
            scene_graph=scene_graph,
            domain_name=intent.domain,
            template_name=intent.style_preference,
            aspect_ratio=intent.aspect_ratio
        )

        # Step 8 & 9: Consistency Validator & Auto-Refinement Loop
        logger.info("Step 8 & 9: Validating content vs prompt consistency")
        consistency_report: ConsistencyReport = self.validator.validate(
            content_analysis=content_analysis,
            prompt_spec=prompt_spec
        )

        attempt = 0
        while not consistency_report.is_consistent and attempt < self.max_refinement_attempts:
            attempt += 1
            logger.info(f"Refinement Pass {attempt}: Prompt failed validation threshold ({consistency_report.overall_score}). Applying feedback: {consistency_report.refinement_feedback}")

            # Refine prompt by adding missing elements
            revised_positive = prompt_spec.positive_prompt + f" Ensure explicit visual emphasis on: {', '.join(consistency_report.discrepancies)}."
            prompt_spec.positive_prompt = revised_positive
            prompt_spec.applied_rules.append(f"Auto-Refinement Pass {attempt}")

            consistency_report = self.validator.validate(
                content_analysis=content_analysis,
                prompt_spec=prompt_spec
            )

        # Step 10: Image Generator (Optional call to provider)
        img_b64 = None
        provider_used = None
        if generate_image_bytes and self.image_service:
            logger.info("Step 10: Dispatching compiled prompt to Image Service")
            try:
                img_bytes = self.image_service.generate_image(
                    prompt=prompt_spec.positive_prompt,
                    negative_prompt=prompt_spec.negative_prompt
                )
                if img_bytes:
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    provider_used = self.image_service.__class__.__name__
            except Exception as e:
                logger.error(f"Image generation failed: {e}")

        return VisualEngineOutput(
            intent=intent,
            generated_content=generated_content,
            content_analysis=content_analysis,
            scene_graph=scene_graph,
            prompt_spec=prompt_spec,
            consistency_report=consistency_report,
            image_bytes_base64=img_b64,
            image_provider_used=provider_used
        )

    def _generate_text_content(self, intent: UserIntent, retrieved_context: str = "") -> ContentGenerationOutput:
        """Produces story/lesson content as the single source of truth (100-180 words, incorporating Human Performance Tools)."""
        if self.llm_service:
            # Load system prompt from prompts/content_generation_system.txt if available
            import os
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prompts")
            sys_prompt_path = os.path.join(prompts_dir, "content_generation_system.txt")
            
            system_prompt = "You are an expert technical content creator and safety coach. Write realistic, impactful learning content (100-180 words). Weave in Human Performance Tools naturally (e.g. Rate Your State (RYS), Anticipating Error, Close Calls, Habit Reminder, RYS Supervisor Conversation)."
            if os.path.exists(sys_prompt_path):
                try:
                    with open(sys_prompt_path, "r", encoding="utf-8") as f:
                        system_prompt = f.read()
                except Exception as exc:
                    logger.warning(f"Could not load content_generation_system.txt: {exc}")

            user_prompt = (
                f"Content Type: Scenario\n"
                f"Topic: {intent.raw_request}\n"
                f"Target Domain: {intent.domain}\n"
                f"Communication Purpose: {intent.communication_purpose}\n"
                f"Retrieved Context: {retrieved_context}\n\n"
                f"Write a realistic 100-180 word scenario. "
                f"Make sure to naturally incorporate one or more Human Performance Tools (such as Rate Your State (RYS), Anticipating Error, Close Calls, Habit Reminder, or RYS Supervisor Conversation)."
            )

            raw_text = self.llm_service._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return ContentGenerationOutput(
                raw_content=raw_text,
                title=f"{intent.domain.capitalize()} {intent.communication_purpose.capitalize()} Lesson",
                core_message=intent.raw_request,
                key_takeaways=["Recognize early human performance states", "Apply Human Performance Tools before hazards escalate"]
            )

        # Built-in fallback content (100-180 words, max 2 paragraphs or bullet points based on card type)
        if "heat" in intent.raw_request.lower() or "warehouse" in intent.raw_request.lower() or intent.domain == "warehouse":
            # Formatted cleanly into 2 paragraphs as a learner-facing mobile scenario card
            para1 = (
                "The midday sun was beating down relentlessly on the loading bay floor. Alex had been maneuvering heavy pallets for over three hours in the stifling heat, pushing past his own growing exhaustion. "
                "He glanced across the bay and noticed his coworker staggering slightly while struggling to move a hydraulic pallet jack. Taking a quick moment for a Rate Your State (RYS) check, Alex recognized that severe physical fatigue and heat stress were setting in."
            )
            para2 = (
                "Before he could call out a warning, the coworker collapsed onto the concrete floor while attempting to lift a heavy load. Alex immediately dropped his clipboard and rushed over to support his fallen teammate, elevating his legs and shouting for emergency medical assistance. "
                "Afterwards, the team conducted an RYS Supervisor Conversation and ran an Anticipating Error session to adjust hydration breaks and prevent future heat illness traps."
            )
            text = f"{para1}\n\n{para2}"
            title = "Heat Stress Prevention & Human Performance Tools"
            core_msg = "Recognize heat stress signs and use Rate Your State (RYS) checks before an emergency occurs."
            takeaways = [
                "Perform a Rate Your State (RYS) self-check when working in high heat",
                "Recognize coworker fatigue and staggering before collapse happens",
                "Hold an RYS Supervisor Conversation to adjust shift breaks and prevent hazards"
            ]
        else:
            para1 = (
                f"Exploring {intent.raw_request} within the {intent.domain} domain to ensure operational excellence and safety. "
                f"Team members actively engaged in daily tasks while maintaining high awareness of operational hazards."
            )
            para2 = (
                f"By taking a moment for a Rate Your State (RYS) check before starting high-risk procedures, employees catch fatigue and distractions early. "
                f"Running through an Anticipating Error pre-task plan helps identify potential traps before they turn into critical errors. "
                f"When near-misses occur, logging them through Close Calls allows the entire organization to learn and continuously improve."
            )
            text = f"{para1}\n\n{para2}"
            title = f"Overview of {intent.raw_request}"
            core_msg = f"Key practices and Human Performance Tools for {intent.raw_request}"
            takeaways = ["Understand fundamentals", "Apply Human Performance Tools", "Maintain continuous vigilance"]

        return ContentGenerationOutput(
            raw_content=text,
            title=title,
            core_message=core_msg,
            key_takeaways=takeaways
        )
