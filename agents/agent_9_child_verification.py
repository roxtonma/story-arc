"""
Agent 9: Child Verification with Regeneration
Verifies child shot images using Gemini 2.5 Pro multimodal capabilities.
Can regenerate failed shots with feedback for up to 3 attempts.
Checks both accuracy to description and consistency with parent shot.
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ChildShotsOutput, VerificationResult
from core.image_utils import save_image_with_metadata


class ChildVerificationAgent(BaseAgent):
    """Agent for verifying and regenerating child shot images."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Child Verification Agent."""
        super().__init__(gemini_client, config, "agent_9")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.child_shots_dir = self.assets_dir / "child_shots"
        self.max_verification_retries = config.get("max_retries", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.consistency_threshold = config.get("consistency_threshold", 0.6)
        self.soft_failure_mode = config.get("soft_failure_mode", True)

        # Load template paths from config
        self.verification_template_path = Path(config.get(
            "verification_prompt_file",
            "prompts/agent_9_verification_prompt.txt"
        ))
        self.modifier_template_path = Path(config.get(
            "modifier_prompt_file",
            "prompts/agent_9_prompt_modifier.txt"
        ))
        self.agent8_template_path = Path(config.get(
            "agent_8_prompt_file",
            "prompts/agent_8_prompt.txt"
        ))

        # Validate templates exist
        if not self.verification_template_path.exists():
            raise FileNotFoundError(f"Verification template not found: {self.verification_template_path}")
        if not self.modifier_template_path.exists():
            raise FileNotFoundError(f"Modifier template not found: {self.modifier_template_path}")
        if not self.agent8_template_path.exists():
            raise FileNotFoundError(f"Agent 8 template not found: {self.agent8_template_path}")

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["child_shots", "parent_shots", "scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        return True

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Verify (and regenerate if needed) child shot images."""
        logger.info(f"{self.agent_name}: Verifying child shot images...")

        child_shots = input_data["child_shots"]
        parent_shots = input_data["parent_shots"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Build comprehensive image lookup including BOTH parent and child shots
        # This is critical for grandchildren (child-of-child) to find their immediate parent
        parent_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        # Add child shots to lookup (for grandchildren support)
        for child in child_shots:
            child_id = child["shot_id"]
            child_path = self.session_dir / child["image_path"]
            parent_images_by_id[child_id] = child_path

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Find parent_id for each child
        child_to_parent = self._map_child_to_parent(shot_grouping)

        # Verify each child shot
        verified_shots = []

        for child_shot in child_shots:
            shot_id = child_shot["shot_id"]

            try:
                logger.info(f"Verifying child shot: {shot_id}")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}")
                    continue

                # Get parent shot ID
                parent_id = child_to_parent.get(shot_id)
                parent_image_path = parent_images_by_id.get(parent_id) if parent_id else None

                # Verify with regeneration capability
                verification_result = self._verify_and_regenerate(
                    child_shot,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars,
                    parent_id,
                    shots_by_id
                )

                # Update child shot data
                child_shot["verification_status"] = verification_result["status"]
                child_shot["final_verification"] = verification_result["final_result"]
                child_shot["verification_history"] = verification_result["history"]
                child_shot["attempts"] = verification_result["attempts"]

                # Update image path if regenerated
                if "new_image_path" in verification_result:
                    child_shot["image_path"] = verification_result["new_image_path"]
                    # CRITICAL: Update lookup so grandchildren use the NEW regenerated path
                    new_full_path = self.session_dir / verification_result["new_image_path"]
                    parent_images_by_id[shot_id] = new_full_path
                    logger.debug(f"Updated parent_images_by_id with regenerated path for {shot_id}")

                verified_shots.append(child_shot)

                status = verification_result["status"]
                logger.info(f"✓ Verification for {shot_id}: {status}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    child_shot["verification_status"] = "soft_failure"
                    child_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [{"category": "Execution Error", "description": str(e)}],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(child_shot)
                    logger.warning(f"Soft failure for {shot_id}, continuing...")
                else:
                    raise

        # Prepare output
        output = {
            "child_shots": verified_shots,
            "total_child_shots": len(verified_shots),
            "metadata": {
                "session_id": self.session_dir.name,
                "verified_at": datetime.now().isoformat(),
                "total_verified": sum(1 for s in verified_shots if s["verification_status"] == "verified"),
                "total_soft_failures": sum(1 for s in verified_shots if s["verification_status"] == "soft_failure")
            }
        }

        logger.info(
            f"{self.agent_name}: Verified {len(verified_shots)} child shots "
            f"({output['metadata']['total_verified']} approved, "
            f"{output['metadata']['total_soft_failures']} soft failures)"
        )

        return output

    def _map_child_to_parent(self, shot_grouping: Dict[str, Any]) -> Dict[str, str]:
        """Map child shot IDs to parent shot IDs."""
        mapping = {}

        def recurse(grouped_shot):
            parent_id = grouped_shot["shot_id"]
            for child in grouped_shot.get("child_shots", []):
                mapping[child["shot_id"]] = parent_id
                recurse(child)

        for parent in shot_grouping.get("parent_shots", []):
            recurse(parent)

        return mapping

    def _verify_and_regenerate(
        self,
        child_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path],
        parent_id: Optional[str],
        shots_by_id: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify child image and regenerate with feedback if verification fails."""
        from google.genai import types

        shot_id = child_shot["shot_id"]
        image_path = self.session_dir / child_shot["image_path"]

        # Get parent shot details and edit type for verification context
        parent_shot_details = shots_by_id.get(parent_id) if parent_id else None

        # Detect edit type
        if parent_shot_details:
            parent_chars = set(parent_shot_details.get("characters", []))
            child_chars = set(shot_details.get("characters", []))

            if len(child_chars) > len(parent_chars):
                edit_type = "add_character"
            elif len(child_chars) < len(parent_chars):
                edit_type = "remove_character"
            elif parent_chars != child_chars:
                edit_type = "character_swap"
            else:
                first_frame_lower = shot_details.get("first_frame", "").lower()
                if any(word in first_frame_lower for word in ["close-up", "close up", "zoom", "tighter"]):
                    edit_type = "camera_change"
                else:
                    edit_type = "expression_change"
        else:
            edit_type = "unknown"

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                # Load child image
                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                child_image = Image.open(current_image_path)

                # Load parent image if available
                parent_image = None
                if parent_image_path and parent_image_path.exists():
                    parent_image = Image.open(parent_image_path)

                # Verify
                verification_prompt = self._format_verification_prompt(
                    shot_details,
                    parent_shot_details,
                    edit_type,
                    has_parent=parent_image is not None
                )

                contents = [child_image]
                if parent_image:
                    contents.append(parent_image)
                contents.append(verification_prompt)

                response = self.client.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        response_json_schema=VerificationResult.model_json_schema()
                    )
                )

                # Check if response.text is None (API failure, safety filter, etc.)
                if response.text is None:
                    raise ValueError(
                        "Gemini API returned empty response (response.text is None). "
                        "This may be due to safety filters, rate limiting, or API errors."
                    )

                # Parse verification result
                result = self._parse_verification_response(response.text)
                verification_history.append(result)

                # Track best result
                if result["confidence"] > best_confidence:
                    best_confidence = result["confidence"]
                    best_result = result

                # Check if approved
                if result["approved"] and result["confidence"] >= self.confidence_threshold:
                    logger.info(f"{shot_id}: Approved (confidence: {result['confidence']})")

                    return_data = {
                        "status": "verified",
                        "final_result": result,
                        "history": verification_history,
                        "attempts": attempt + 1
                    }

                    if attempt > 0:
                        try:
                            return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
                        except ValueError:
                            return_data["new_image_path"] = str(current_image_path)

                    return return_data

                # Not approved - regenerate for next attempt
                if attempt < self.max_verification_retries - 1:
                    logger.warning(f"{shot_id}: Not approved, regenerating...")

                    # Regenerate with intelligent prompt rewriting
                    new_image_path = self._regenerate_child_shot(
                        shot_id,
                        shot_details,
                        scene_breakdown,
                        parent_image_path,
                        grids_by_chars,
                        result.get("issues", []),
                        edit_type,
                        attempt + 1,
                        child_shot  # Pass original child shot data with saved prompts
                    )

                    current_image_path = new_image_path
                    logger.info(f"Regenerated {shot_id}, will verify new image")

            except Exception as e:
                logger.error(f"Verification attempt {attempt + 1} failed: {str(e)}")
                verification_history.append({
                    "approved": False,
                    "confidence": 0.0,
                    "issues": [{"category": "Verification Error", "description": str(e)}],
                    "recommendation": "regenerate"
                })

        # All attempts failed - soft failure
        logger.warning(f"{shot_id}: All verification attempts failed, soft failure")

        fallback_result = {
            "approved": False,
            "confidence": 0.0,
            "issues": [{"category": "Verification Failure", "description": "All verification attempts failed"}],
            "recommendation": "manual_review"
        }

        return_data = {
            "status": "soft_failure",
            "final_result": best_result or (verification_history[-1] if verification_history else fallback_result),
            "history": verification_history,
            "attempts": self.max_verification_retries
        }

        if current_image_path != image_path:
            try:
                return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
            except ValueError:
                return_data["new_image_path"] = str(current_image_path)

        return return_data

    def _regenerate_child_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path],
        verification_issues: List[Dict[str, str]],
        edit_type: str,
        attempt_number: int,
        original_child_shot: Dict[str, Any]
    ) -> Path:
        """Regenerate child shot with intelligently rewritten prompt based on verification issues."""
        from google.genai import types

        # Validate parent image exists
        if not parent_image_path:
            raise ValueError(
                f"Parent image path is None for shot {shot_id}. "
                "This shot may be a grandchild whose parent is not in the parent_images lookup. "
                "Check that the child shot's parent was successfully generated."
            )

        if not parent_image_path.exists():
            raise ValueError(
                f"Parent image required for child regeneration but file not found: {parent_image_path}. "
                f"Shot: {shot_id}"
            )

        # Load parent image
        parent_image = Image.open(parent_image_path)

        # Find grid if needed
        characters = shot_details.get("characters", [])
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        # Step 1: Load metadata from disk JSON file to get original optimized prompt
        image_path = self.session_dir / original_child_shot["image_path"]
        metadata_path = image_path.with_suffix('.json')

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for retry: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        original_optimized_prompt = metadata.get("prompts", {}).get("optimized_prompt")

        if not original_optimized_prompt:
            logger.warning(f"{shot_id}: No optimized prompt in metadata")
            raise ValueError(f"Original optimized prompt not found in metadata for {shot_id}")

        logger.debug(f"{shot_id}: Loaded optimized prompt from disk metadata: {metadata_path.name}")

        # Step 2: Intelligently rewrite the OPTIMIZED prompt (not verbose!)
        rewritten_prompt = self._rewrite_prompt_with_feedback(
            original_optimized_prompt,  # Rewrite the actual prompt that was sent to Flash
            verification_issues,
            shot_details,
            edit_type,
            shot_id
        )

        # Step 3: Use rewritten prompt directly (no additional optimization needed)
        final_prompt = rewritten_prompt

        # Prepare contents
        contents = [parent_image]
        if grid_path and grid_path.exists():
            grid_image = Image.open(grid_path)
            contents.append(grid_image)
        contents.append(final_prompt)

        # Generate
        response = self.client.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                temperature=0.8,
            ),
        )

        # Extract and save
        generated_image = None
        for part in response.parts:
            if part.inline_data is not None:
                gemini_image = part.as_image()
                generated_image = Image.open(BytesIO(gemini_image.image_bytes))
                break

        if not generated_image:
            raise ValueError(f"No image generated for {shot_id} retry")

        # Save with retry suffix
        image_filename = f"{shot_id}_child_retry{attempt_number}.png"
        new_image_path = self.child_shots_dir / image_filename

        save_image_with_metadata(generated_image, new_image_path, metadata={
            "shot_id": shot_id,
            "attempt": attempt_number,
            "edit_type": edit_type,
            "grid_used": str(grid_path) if grid_path and grid_path.exists() else None,
            "regenerated_with_feedback": verification_issues,
            "prompts": {
                "original_optimized_prompt": original_optimized_prompt,
                "rewritten_prompt": rewritten_prompt
            }
        })

        logger.info(f"Regenerated {shot_id}: {new_image_path.name}")
        return new_image_path

    def _get_character_physical_descriptions(self, character_names: List[str], scene_breakdown: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract full physical descriptions for characters."""
        descriptions = []
        char_lookup = {}
        for scene in scene_breakdown.get("scenes", []):
            for char in scene.get("characters", []):
                if char.get("name"):
                    char_lookup[char["name"]] = char.get("description", "")

        for name in character_names:
            if name in char_lookup:
                descriptions.append({"name": name, "physical_description": char_lookup[name]})
            else:
                descriptions.append({"name": name, "physical_description": f"Character named {name}"})

        return descriptions

    def _analyze_failure_pattern(
        self,
        issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        edit_type: str
    ) -> str:
        """
        Analyze issues to identify failure patterns and suggest rewrite strategy.

        Args:
            issues: List of categorized issues from verification
            shot_details: Child shot details
            edit_type: Type of edit operation

        Returns:
            Pattern analysis string for prompt modifier
        """
        categories = [issue.get("category", "") for issue in issues]

        patterns = []

        # OTS failure pattern
        first_frame_upper = shot_details.get("first_frame", "").upper()
        if "OTS Failure" in categories or ("Extra Character" in categories and any(ots in first_frame_upper for ots in ["OVER-THE-SHOULDER", "OVER THE SHOULDER", "OTS"])):
            patterns.append("OTS_SHOT_MISINTERPRETED: Model created new character instead of changing camera angle. Needs explicit spatial frame description with foreground/background positioning.")

        # Character duplication or extras
        if "Extra Character" in categories or "Duplicate Character" in categories:
            char_count = len(shot_details.get("characters", []))
            patterns.append(f"EXTRA_CHARACTER: More than {char_count} characters appeared. Needs explicit 'EXACTLY {char_count} characters' constraint.")

        # Poor integration
        if "Pasted Character" in categories:
            patterns.append("POOR_INTEGRATION: Added character looks artificial. Needs lighting direction/intensity match, scale/depth matching, and spatial anchoring instructions.")

        # Grid artifacts
        if "Grid Artifact" in categories:
            patterns.append("GRID_ARTIFACT: Reference grid structure leaked into output. Needs strong 'unified scene, no grid elements' emphasis at prompt start.")

        # Composition issues
        if "Bad Composition" in categories:
            patterns.append("COMPOSITION_ISSUE: Vague framing instructions caused awkward layout. Needs objective spatial description of frame regions.")

        # Proportion issues
        if "Bad Proportions" in categories:
            patterns.append("PROPORTION_ISSUE: Character scales inconsistent. Needs explicit scale/distance references and perspective cues.")

        # Missing characters
        if "Missing Character" in categories:
            patterns.append("MISSING_CHARACTER: Expected character not visible. Needs explicit placement instruction for each character.")

        return "\n".join(f"- {pattern}" for pattern in patterns) if patterns else "- GENERAL_ISSUES: Address the specific issues listed."

    def _rewrite_prompt_with_feedback(
        self,
        original_prompt: str,
        categorized_issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        edit_type: str,
        shot_id: str
    ) -> str:
        """
        Use Pro 2.5 to intelligently rewrite prompt based on verification issues.

        Args:
            original_prompt: The verbose prompt that was used
            categorized_issues: List of issues with category and description
            shot_details: Child shot target details
            edit_type: Type of edit operation
            shot_id: Shot identifier for logging

        Returns:
            Rewritten prompt addressing root causes
        """
        from google.genai import types

        # Load modifier template from config
        modifier_template_path = self.modifier_template_path
        if not modifier_template_path.exists():
            logger.warning(f"{shot_id}: Modifier template not found, using feedback append fallback")
            # Fallback: just append issues
            issues_text = "\n".join(f"- {issue['category']}: {issue['description']}" for issue in categorized_issues)
            return original_prompt + f"\n\nCRITICAL ISSUES TO FIX:\n{issues_text}"

        with open(modifier_template_path, 'r', encoding='utf-8') as f:
            modifier_template = f.read()

        # Analyze failure patterns
        pattern_analysis = self._analyze_failure_pattern(categorized_issues, shot_details, edit_type)

        # Format issues for template
        issues_text = "\n".join(
            f"- Category: {issue['category']}\n  Description: {issue['description']}"
            for issue in categorized_issues
        )

        # Create modification prompt
        modification_prompt = modifier_template.format(
            original_prompt=original_prompt,
            categorized_issues=issues_text,
            pattern_analysis=pattern_analysis
        )

        try:
            logger.debug(f"{shot_id}: Rewriting prompt with Pro 2.5 based on patterns...")
            response = self.client.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[modification_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192
                )
            )

            if response.text:
                rewritten_prompt = response.text.strip()
                logger.info(f"{shot_id}: Prompt intelligently rewritten (patterns: {len(pattern_analysis.split(chr(10)))} detected)")
                return rewritten_prompt
            else:
                logger.warning(f"{shot_id}: Prompt rewrite returned empty, using original")
                return original_prompt

        except Exception as e:
            logger.error(f"{shot_id}: Prompt rewrite failed: {str(e)}, using original")
            return original_prompt

    def _format_verification_prompt(
        self,
        shot_details: Dict[str, Any],
        parent_shot_details: Optional[Dict[str, Any]],
        edit_type: str,
        has_parent: bool
    ) -> str:
        """
        Format verification prompt from template with full requirement context.

        Args:
            shot_details: Child shot target details
            parent_shot_details: Parent shot details (for comparison)
            edit_type: Type of edit operation
            has_parent: Whether parent image is available

        Returns:
            Formatted verification prompt
        """
        # Load verification template from config
        template_path = self.verification_template_path
        if not template_path.exists():
            logger.warning(f"Verification template not found: {template_path}")
            raise FileNotFoundError(f"Verification template required but not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Prepare context
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location = shot_details.get("location", "")

        # Parent context
        parent_context_prefix = " and compare it with the parent shot image (if provided)" if has_parent else ""

        if has_parent and parent_shot_details:
            parent_first_frame = parent_shot_details.get("first_frame", "")
            parent_characters = parent_shot_details.get("characters", [])
            parent_context = f"""
PARENT SHOT CONTEXT (What we're editing from):
- Characters in parent: {", ".join(parent_characters)}
- Parent composition: {parent_first_frame}

NOTE: If the edit type is '{edit_type}', then differences in characters/composition may be INTENTIONAL.
"""
        else:
            parent_context = ""

        # Format template
        prompt = template.format(
            parent_context_prefix=parent_context_prefix,
            expected_characters=", ".join(characters),
            location=location,
            first_frame=first_frame,
            edit_type=edit_type,
            parent_context=parent_context
        )

        return prompt

    def _parse_verification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse verification JSON response with schema validation."""
        try:
            result = json.loads(response_text)
            # Validate with Pydantic
            validated = VerificationResult(**result)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Verification response parse/validation failed: {str(e)}")
            return {
                "approved": False,
                "confidence": 0.0,
                "issues": [{"category": "Parse Error", "description": str(e)}],
                "recommendation": "manual_review"
            }
