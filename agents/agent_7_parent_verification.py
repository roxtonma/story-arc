"""
Agent 7: Parent Verification with Regeneration
Verifies parent shot images using Gemini 2.5 Pro multimodal capabilities.
Can regenerate failed shots with feedback for up to 3 attempts.
"""

import json
from io import BytesIO
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ParentShotsOutput, VerificationResult
from core.image_utils import save_image_with_metadata


class ParentVerificationAgent(BaseAgent):
    """Agent for verifying and regenerating parent shot images."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Parent Verification Agent."""
        super().__init__(gemini_client, config, "agent_7")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.parent_shots_dir = self.assets_dir / "parent_shots"
        self.max_verification_retries = config.get("max_retries", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.soft_failure_mode = config.get("soft_failure_mode", True)

        # Load template paths from config
        self.verification_template_path = Path(config.get(
            "verification_prompt_file",
            "prompts/agent_7_verification_prompt.txt"
        ))
        self.modifier_template_path = Path(config.get(
            "modifier_prompt_file",
            "prompts/agent_7_prompt_modifier.txt"
        ))
        self.agent6_template_path = Path(config.get(
            "agent_6_prompt_file",
            "prompts/agent_6_prompt.txt"
        ))

        # Validate templates exist
        if not self.verification_template_path.exists():
            raise FileNotFoundError(f"Verification template not found: {self.verification_template_path}")
        if not self.modifier_template_path.exists():
            raise FileNotFoundError(f"Modifier template not found: {self.modifier_template_path}")
        if not self.agent6_template_path.exists():
            raise FileNotFoundError(f"Agent 6 template not found: {self.agent6_template_path}")

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["parent_shots", "scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
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
        """Verify (and regenerate if needed) parent shot images."""
        logger.info(f"{self.agent_name}: Verifying parent shot images...")

        parent_shots = input_data["parent_shots"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): grid["grid_path"]
            for grid in character_grids
        }

        # Verify each parent shot
        verified_shots = []

        for parent_shot in parent_shots:
            shot_id = parent_shot["shot_id"]

            try:
                logger.info(f"Verifying parent shot: {shot_id}")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}")
                    continue

                # Verify with regeneration capability
                verification_result = self._verify_and_regenerate(
                    parent_shot,
                    shot_details,
                    scene_breakdown,
                    grids_by_chars
                )

                # Update parent shot data
                parent_shot["verification_status"] = verification_result["status"]
                parent_shot["final_verification"] = verification_result["final_result"]
                parent_shot["verification_history"] = verification_result["history"]
                parent_shot["attempts"] = verification_result["attempts"]
                # Update image path if regenerated
                if "new_image_path" in verification_result:
                    parent_shot["image_path"] = verification_result["new_image_path"]

                verified_shots.append(parent_shot)

                status = verification_result["status"]
                logger.info(f"✓ Verification for {shot_id}: {status}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    parent_shot["verification_status"] = "soft_failure"
                    parent_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [{"category": "Execution Error", "description": str(e)}],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(parent_shot)
                    logger.warning(f"Soft failure for {shot_id}, continuing...")
                else:
                    raise

        # Prepare output
        output = {
            "parent_shots": verified_shots,
            "total_parent_shots": len(verified_shots),
            "metadata": {
                "session_id": self.session_dir.name,
                "verified_at": datetime.now().isoformat(),
                "total_verified": sum(1 for s in verified_shots if s["verification_status"] == "verified"),
                "total_soft_failures": sum(1 for s in verified_shots if s["verification_status"] == "soft_failure")
            }
        }

        logger.info(
            f"{self.agent_name}: Verified {len(verified_shots)} parent shots "
            f"({output['metadata']['total_verified']} approved, "
            f"{output['metadata']['total_soft_failures']} soft failures)"
        )

        return output

    def _verify_and_regenerate(
        self,
        parent_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str]
    ) -> Dict[str, Any]:
        """Verify image and regenerate with feedback if verification fails."""
        from google.genai import types

        shot_id = parent_shot["shot_id"]
        image_path = self.session_dir / parent_shot["image_path"]

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                # Load current image
                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                image = Image.open(current_image_path)

                # Verify image
                verification_prompt = self._format_verification_prompt(shot_details)

                response = self.client.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[image, verification_prompt],
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

                    # Return new path if we regenerated
                    if attempt > 0:
                        try:
                            return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
                        except ValueError:
                            return_data["new_image_path"] = str(current_image_path)

                    return return_data

                # Not approved - regenerate for next attempt
                if attempt < self.max_verification_retries - 1:
                    logger.warning(
                        f"{shot_id}: Not approved (confidence: {result['confidence']}), "
                        f"issues: {result.get('issues', [])}. Regenerating..."
                    )

                    # Regenerate with feedback
                    new_image_path = self._regenerate_parent_shot(
                        shot_id,
                        shot_details,
                        scene_breakdown,
                        grids_by_chars,
                        result.get("issues", []),
                        attempt + 1,
                        parent_shot  # Pass original parent shot data with saved prompts
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

        # Return new path if we regenerated
        if current_image_path != image_path:
            try:
                return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
            except ValueError:
                return_data["new_image_path"] = str(current_image_path)

        return return_data

    def _regenerate_parent_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str],
        verification_issues: List[Dict[str, str]],
        attempt_number: int,
        original_parent_shot: Dict[str, Any]
    ) -> Path:
        """Regenerate parent shot with intelligently rewritten prompt based on verification issues."""
        from google.genai import types

        # Find grid if needed
        characters = shot_details.get("characters", [])
        grid_image = None
        grid_path = None
        if characters:
            char_combo = tuple(sorted(characters))
            grid_path = grids_by_chars.get(char_combo)
            if grid_path:
                full_grid_path = self.session_dir / grid_path
                if full_grid_path.exists():
                    grid_image = Image.open(full_grid_path)

        # Step 1: Load metadata from disk JSON file to get original optimized prompt
        image_path = self.session_dir / original_parent_shot["image_path"]
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
            shot_id
        )

        # Step 3: Use rewritten prompt directly (no additional optimization needed)
        final_prompt = rewritten_prompt

        # Generate
        if grid_image:
            contents = [grid_image, final_prompt]
        else:
            contents = [final_prompt]

        response = self.client.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                temperature=0.8,  # Slightly higher for variation
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
        image_filename = f"{shot_id}_parent_retry{attempt_number}.png"
        new_image_path = self.parent_shots_dir / image_filename

        save_image_with_metadata(generated_image, new_image_path, metadata={
            "shot_id": shot_id,
            "attempt": attempt_number,
            "grid_used": str(grid_path) if grid_path else "none",
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

    def _get_location_full_description(self, location_name: str, scene_id: str, scene_breakdown: Dict[str, Any]) -> str:
        """Extract full location description."""
        for scene in scene_breakdown.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                return scene.get("location", {}).get("description", location_name)
        return location_name

    def _analyze_failure_pattern(
        self,
        issues: List[Dict[str, str]],
        shot_details: Dict[str, Any]
    ) -> str:
        """
        Analyze issues to identify failure patterns for parent shot generation.

        Args:
            issues: List of categorized issues from verification
            shot_details: Parent shot details

        Returns:
            Pattern analysis string for prompt modifier
        """
        categories = [issue.get("category", "") for issue in issues]

        patterns = []

        # Character count issues
        if "Extra Character" in categories or "Duplicate Character" in categories:
            char_count = len(shot_details.get("characters", []))
            patterns.append(f"EXTRA_CHARACTER: More than {char_count} characters appeared. Needs explicit 'EXACTLY {char_count} characters' constraint.")

        # Missing characters
        if "Missing Character" in categories:
            patterns.append("MISSING_CHARACTER: Expected character not visible. Needs explicit placement instruction for each character.")

        # Poor integration
        if "Pasted Character" in categories:
            patterns.append("POOR_INTEGRATION: Character looks artificial. Needs lighting direction/intensity match, scale/depth matching, and natural placement instructions.")

        # Grid artifacts
        if "Grid Artifact" in categories:
            patterns.append("GRID_ARTIFACT: Character grid structure leaked into output. Needs strong 'unified scene, NO grid elements, fill entire frame' emphasis at prompt start.")

        # Composition issues
        if "Bad Composition" in categories:
            patterns.append("COMPOSITION_ISSUE: Vague framing instructions caused awkward layout. Needs objective spatial description of frame regions and character positions.")

        # Proportion issues
        if "Bad Proportions" in categories:
            patterns.append("PROPORTION_ISSUE: Character scales inconsistent. Needs explicit scale/distance references and perspective cues.")

        # Lighting mismatch
        if "Lighting Mismatch" in categories:
            patterns.append("LIGHTING_MISMATCH: Characters have different lighting. Needs unified light source description affecting all characters equally.")

        # Poor location
        if "Poor Location" in categories:
            patterns.append("POOR_LOCATION: Environment lacks detail or doesn't match description. Needs more specific environmental details and architectural elements.")

        return "\n".join(f"- {pattern}" for pattern in patterns) if patterns else "- GENERAL_ISSUES: Address the specific issues listed."

    def _rewrite_prompt_with_feedback(
        self,
        original_prompt: str,
        categorized_issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        shot_id: str
    ) -> str:
        """
        Use Pro 2.5 to intelligently rewrite prompt based on verification issues.

        Args:
            original_prompt: The verbose prompt that was used
            categorized_issues: List of issues with category and description
            shot_details: Parent shot target details
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
        pattern_analysis = self._analyze_failure_pattern(categorized_issues, shot_details)

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

    def _format_verification_prompt(self, shot_details: Dict[str, Any]) -> str:
        """
        Format verification prompt from template with full requirement context.

        Args:
            shot_details: Parent shot target details

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

        # Format template
        prompt = template.format(
            expected_characters=", ".join(characters),
            location=location,
            first_frame=first_frame
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
