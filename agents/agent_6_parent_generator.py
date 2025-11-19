"""
Agent 6: Parent Image Generator
Transforms character grids into full cinematic scenes using Gemini 2.5 Flash Image.
Uses character grids as PRIMARY INPUT to maintain consistency.
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
from core.validators import ParentShotsOutput
from core.image_utils import save_image_with_metadata
from utils.fal_helper import (
    generate_with_fal_text_to_image,
    generate_with_fal_edit,
    is_fal_available
)


class ParentImageGeneratorAgent(BaseAgent):
    """Agent for generating parent shot images by transforming character grids."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Parent Image Generator Agent."""
        super().__init__(gemini_client, config, "agent_6")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.parent_shots_dir = self.assets_dir / "parent_shots"
        self.grids_dir = self.assets_dir / "grids"

        # Create directory
        self.parent_shots_dir.mkdir(parents=True, exist_ok=True)

        # Prompt optimization toggle and template path
        self.use_optimizer = config.get("use_prompt_optimizer", True)
        self.optimizer_template_path = Path(config.get(
            "optimizer_prompt_file",
            "prompts/agent_6_optimizer_prompt.txt"
        ))

        # Validate optimizer template if enabled
        if self.use_optimizer and not self.optimizer_template_path.exists():
            raise FileNotFoundError(f"Optimizer template not found: {self.optimizer_template_path}")

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output parent shots data."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            parent_output = ParentShotsOutput(**output_data)

            if parent_output.total_parent_shots == 0:
                raise ValueError("No parent shots generated")

            # Auto-fix count
            output_data["total_parent_shots"] = len(parent_output.parent_shots)

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Parent shots validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Generate parent shot images."""
        logger.info(f"{self.agent_name}: Transforming character grids into parent shots...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        character_grids = input_data["character_grids"]

        # Create shot lookup
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Create grid lookup
        grids_by_chars = {
            tuple(sorted(grid["characters"])): grid["grid_path"]
            for grid in character_grids
        }

        # Extract parent shots
        parent_shot_ids = self._extract_parent_shots(shot_grouping)
        logger.info(f"Found {len(parent_shot_ids)} parent shots to generate")

        # Generate parent images
        parent_shots_data = []

        for shot_id in parent_shot_ids:
            try:
                logger.info(f"Generating parent shot: {shot_id}")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}, skipping")
                    continue

                # Generate image
                image_path = self._generate_parent_shot_image(
                    shot_id,
                    shot_details,
                    scene_breakdown,
                    grids_by_chars
                )

                # Store data
                # Try to get relative path, fallback to absolute if fails
                try:
                    rel_image_path = str(image_path.relative_to(self.session_dir))
                except ValueError:
                    # Fallback to absolute path (cross-drive on Windows)
                    rel_image_path = str(image_path)

                parent_shots_data.append({
                    "shot_id": shot_id,
                    "scene_id": shot_details.get("scene_id"),
                    "image_path": rel_image_path,
                    "generation_timestamp": datetime.now().isoformat(),
                    "verification_status": "pending",
                    "attempts": 1,
                    "final_verification": None,
                    "verification_history": []
                })

                logger.info(f"✓ Generated parent shot: {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate parent shot {shot_id}: {str(e)}")
                raise

        # Prepare output
        output = {
            "parent_shots": parent_shots_data,
            "total_parent_shots": len(parent_shots_data),
            "metadata": {
                "session_id": self.session_dir.name,
                "generated_at": datetime.now().isoformat()
            }
        }

        logger.info(f"{self.agent_name}: Generated {len(parent_shots_data)} parent shots")

        return output

    def _extract_parent_shots(self, shot_grouping: Dict[str, Any]) -> List[str]:
        """Extract parent shot IDs from grouping."""
        parent_ids = []

        for parent_shot in shot_grouping.get("parent_shots", []):
            parent_ids.append(parent_shot["shot_id"])

        return parent_ids

    def _get_character_physical_descriptions(
        self,
        character_names: List[str],
        scene_breakdown: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Extract full physical descriptions for characters from Agent 2 scene breakdown.

        Args:
            character_names: List of character names in the shot
            scene_breakdown: Agent 2 output with verbose character descriptions

        Returns:
            List of dicts with name and full physical description
        """
        descriptions = []

        # Build character lookup from all scenes
        char_lookup = {}
        for scene in scene_breakdown.get("scenes", []):
            for char in scene.get("characters", []):
                char_name = char.get("name")
                char_desc = char.get("description", "")
                if char_name and char_desc:
                    char_lookup[char_name] = char_desc

        # Get descriptions for requested characters
        for name in character_names:
            if name in char_lookup:
                descriptions.append({
                    "name": name,
                    "physical_description": char_lookup[name]
                })
            else:
                logger.warning(f"Physical description not found for character: {name}")
                descriptions.append({
                    "name": name,
                    "physical_description": f"Character named {name}"
                })

        return descriptions

    def _get_location_full_description(
        self,
        location_name: str,
        scene_id: str,
        scene_breakdown: Dict[str, Any]
    ) -> str:
        """
        Extract full location description from Agent 2 scene breakdown.

        Args:
            location_name: Location name
            scene_id: Scene identifier
            scene_breakdown: Agent 2 output

        Returns:
            Full verbose location description
        """
        # Try to find the exact scene
        for scene in scene_breakdown.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                location = scene.get("location", {})
                return location.get("description", location_name)

        # Fallback: search by location name
        for scene in scene_breakdown.get("scenes", []):
            location = scene.get("location", {})
            if location.get("name") == location_name:
                return location.get("description", location_name)

        logger.warning(f"Full location description not found for: {location_name}")
        return location_name

    def _optimize_prompt_with_pro(self, verbose_prompt: str) -> str:
        """
        Use Gemini Pro 2.5 to optimize the verbose prompt for image generation.

        Args:
            verbose_prompt: The filled template prompt

        Returns:
            Optimized prompt for Flash Image
        """
        from google.genai import types

        # Check if optimization is enabled
        if not self.use_optimizer:
            logger.debug("Prompt optimization disabled in config")
            return verbose_prompt

        # Load optimizer template from config
        optimizer_template_path = self.optimizer_template_path
        if not optimizer_template_path.exists():
            logger.warning(f"Optimizer template not found: {optimizer_template_path}, using verbose prompt as-is")
            return verbose_prompt

        with open(optimizer_template_path, 'r', encoding='utf-8') as f:
            optimizer_template = f.read()

        # Create optimization prompt
        optimization_prompt = optimizer_template.format(
            verbose_scene_description=verbose_prompt
        )

        try:
            # Use Pro 2.5 to optimize
            logger.debug("Optimizing prompt with Gemini Pro 2.5...")
            response = self.client.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[optimization_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Low temp for consistent optimization
                    max_output_tokens=8192
                )
            )

            if response.text:
                optimized = response.text.strip()
                logger.info(f"Prompt optimized: {len(verbose_prompt)} → {len(optimized)} chars")
                return optimized
            else:
                logger.warning("Pro optimization returned empty, using verbose prompt")
                return verbose_prompt

        except Exception as e:
            logger.warning(f"Prompt optimization failed: {str(e)}, using verbose prompt")
            return verbose_prompt

    def _generate_parent_shot_image(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str]
    ) -> Path:
        """
        Transform character grid into full parent shot image.
        Uses grid as PRIMARY INPUT for character consistency.
        """
        from google.genai import types

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        scene_id = shot_details.get("scene_id", "")
        location_name = shot_details.get("location", "")
        dialogue = shot_details.get("dialogue", "")

        # Get verbose character descriptions (PHYSICAL TRAITS, not names!)
        character_descriptions = self._get_character_physical_descriptions(
            characters,
            scene_breakdown
        )

        # Get verbose location description
        location_description = self._get_location_full_description(
            location_name,
            scene_id,
            scene_breakdown
        )

        # Find matching character grid (OPTIONAL - only needed if shot has characters)
        grid_image = None
        grid_path = None

        if characters:
            # Shot has characters - require character grid
            char_combo = tuple(sorted(characters))
            grid_path = grids_by_chars.get(char_combo)

            if not grid_path:
                raise ValueError(
                    f"No character grid found for combination: {characters}. "
                    "Cannot generate shot with characters without their grid."
                )

            full_grid_path = self.session_dir / grid_path
            if not full_grid_path.exists():
                raise FileNotFoundError(f"Character grid file not found: {full_grid_path}")

            # Load grid as PRIMARY INPUT
            grid_image = Image.open(full_grid_path)
            logger.info(f"Transforming character grid: {grid_path}")
        else:
            # No characters in shot - generate without grid (establishing shot, empty scene, etc.)
            logger.info(f"Shot {shot_id} has no characters - generating without grid")

        # Format character descriptions into template string
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""
CHARACTER {idx} (from grid position {idx}):
PHYSICAL APPEARANCE: {char['physical_description']}

This character must be placed in the scene exactly as they appear in the grid, maintaining their physical appearance, clothing, hair, facial features, and all other visual characteristics precisely.

"""

        # Use template from prompt file
        if not self.prompt_template:
            raise ValueError("Prompt template not loaded")

        verbose_prompt = self.prompt_template.format(
            shot_id=shot_id,
            first_frame=first_frame,
            location_description=location_description,
            character_descriptions=char_desc_text
        )

        # Optimize prompt using Pro 2.5 before sending to Flash Image (or fal)
        optimized_prompt = self._optimize_prompt_with_pro(verbose_prompt)

        # Check image provider from config
        image_provider = self.config.get("image_provider", "gemini").lower()
        fal_seed = None

        if image_provider == "fal":
            # Use fal for image generation
            logger.info(f"Using fal for parent shot generation: {shot_id}")

            if not is_fal_available():
                logger.warning("fal_client not available, falling back to Gemini")
                image_provider = "gemini"
            else:
                try:
                    if grid_image:
                        # Use fal edit model with grid as input (transformation)
                        fal_model = self.config.get(
                            "fal_edit_model",
                            "fal-ai/bytedance/seedream/v4/edit"
                        )
                        logger.debug("Using fal edit mode with grid transformation")

                        # Save grid temporarily to upload to fal
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            grid_image.save(tmp.name)
                            tmp_grid_path = Path(tmp.name)

                        try:
                            generated_image, fal_seed = generate_with_fal_edit(
                                prompt=optimized_prompt,
                                image_paths=[tmp_grid_path],
                                model=fal_model,
                                width=3840,
                                height=2160,
                                num_images=1,
                                enable_safety_checker=True,
                                enhance_prompt_mode="standard"
                            )
                        finally:
                            # Clean up temp file
                            tmp_grid_path.unlink(missing_ok=True)

                    else:
                        # Use fal text-to-image model (no grid)
                        fal_model = self.config.get(
                            "fal_text_to_image_model",
                            "fal-ai/bytedance/seedream/v4/text-to-image"
                        )
                        logger.debug("Using fal text-to-image mode")

                        generated_image, fal_seed = generate_with_fal_text_to_image(
                            prompt=optimized_prompt,
                            model=fal_model,
                            width=3840,
                            height=2160,
                            num_images=1,
                            enable_safety_checker=True,
                            enhance_prompt_mode="standard"
                        )

                    logger.info(f"Successfully generated parent shot with fal (seed: {fal_seed})")

                except Exception as e:
                    logger.error(f"Failed to generate with fal: {e}, falling back to Gemini")
                    image_provider = "gemini"
                    generated_image = None

        # Use Gemini for image generation (default or fallback)
        if image_provider == "gemini":
            from google.genai import types

            logger.info(f"Using Gemini for parent shot generation: {shot_id}")

            # Prepare contents: Grid first if available (transformation), otherwise text only (generation)
            if grid_image:
                # CRITICAL: Grid image is FIRST (the thing being transformed)
                contents = [grid_image, optimized_prompt]
                logger.debug("Using grid transformation mode with optimized prompt")
            else:
                # No grid - generate from description only
                contents = [optimized_prompt]
                logger.debug("Using text-only generation mode with optimized prompt")

            # Generate/transform image
            response = self.client.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                    ),
                    temperature=self.temperature,
                ),
            )

            # Extract generated image and convert to PIL Image
            generated_image = None
            for part in response.parts:
                if part.inline_data is not None:
                    # Get Gemini SDK Image wrapper
                    gemini_image = part.as_image()
                    # Convert to PIL Image
                    generated_image = Image.open(BytesIO(gemini_image.image_bytes))
                    break

        if not generated_image:
            raise ValueError(f"No image generated for {shot_id}")

        # Save image
        image_filename = f"{shot_id}_parent.png"
        image_path = self.parent_shots_dir / image_filename

        metadata = {
            "shot_id": shot_id,
            "characters": characters,
            "location": location_name,
            "generated_at": datetime.now().isoformat(),
            "grid_used": str(grid_path) if grid_path else "none",
            "image_provider": image_provider,
            "prompts": {
                "verbose_prompt": verbose_prompt,
                "optimized_prompt": optimized_prompt,
                "optimizer_used": self.use_optimizer
            }
        }

        # Add fal-specific metadata if applicable
        if fal_seed is not None:
            metadata["fal_seed"] = fal_seed

        save_image_with_metadata(
            generated_image,
            image_path,
            metadata=metadata
        )

        return image_path
