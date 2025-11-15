"""
Agent 8: Child Image Generator
Generates child shot images by editing parent shots using Gemini 2.5 Flash Image.
Uses physical trait descriptions and character grids for consistency.
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
from core.validators import ChildShotsOutput
from core.image_utils import save_image_with_metadata


class ChildImageGeneratorAgent(BaseAgent):
    """Agent for generating child shot images by editing parent shots."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Child Image Generator Agent."""
        super().__init__(gemini_client, config, "agent_8")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.child_shots_dir = self.assets_dir / "child_shots"
        self.parent_shots_dir = self.assets_dir / "parent_shots"
        self.grids_dir = self.assets_dir / "grids"

        # Create directory
        self.child_shots_dir.mkdir(parents=True, exist_ok=True)

        # Prompt optimization toggle
        self.use_optimizer = config.get("use_prompt_optimizer", True)

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["scene_breakdown", "shot_breakdown", "shot_grouping", "parent_shots", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            child_output = ChildShotsOutput(**output_data)

            # Auto-fix count
            output_data["total_child_shots"] = len(child_output.child_shots)

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Child shots validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Generate child shot images."""
        logger.info(f"{self.agent_name}: Editing parent shots to create child shots...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        parent_shots = input_data["parent_shots"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Dynamic image lookup - starts with top-level parents, grows as children are generated
        # This allows grandchildren to use their immediate parent (which is a child shot)
        available_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Extract all child shots
        child_shot_list = self._extract_child_shots(shot_grouping)
        logger.info(f"Found {len(child_shot_list)} child shots to generate")

        # Generate child images
        child_shots_data = []

        for child_shot_info in child_shot_list:
            try:
                shot_id = child_shot_info["shot_id"]
                parent_id = child_shot_info["parent_shot_id"]

                logger.info(f"Generating child shot: {shot_id} (parent: {parent_id})")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}, skipping")
                    continue

                # Get parent image from dynamic lookup (includes previously generated children)
                parent_image_path = available_images_by_id.get(parent_id)
                if not parent_image_path or not parent_image_path.exists():
                    logger.warning(f"Parent image not found for {parent_id}, skipping")
                    continue

                # Generate child image
                child_image_path = self._generate_child_shot_image(
                    shot_id,
                    parent_id,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars,
                    shots_by_id
                )

                # Add newly generated child to lookup for future grandchildren
                available_images_by_id[shot_id] = child_image_path

                # Store data
                # Try to get relative path, fallback to absolute if fails
                try:
                    rel_image_path = str(child_image_path.relative_to(self.session_dir))
                except ValueError:
                    # Fallback to absolute path (cross-drive on Windows)
                    rel_image_path = str(child_image_path)

                child_shots_data.append({
                    "shot_id": shot_id,
                    "scene_id": shot_details.get("scene_id"),
                    "image_path": rel_image_path,
                    "generation_timestamp": datetime.now().isoformat(),
                    "verification_status": "pending",
                    "attempts": 1,
                    "final_verification": None,
                    "verification_history": []
                })

                logger.info(f"✓ Generated child shot: {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate child shot {shot_id}: {str(e)}")
                # Continue with other shots (soft failure)
                continue

        # Prepare output
        output = {
            "child_shots": child_shots_data,
            "total_child_shots": len(child_shots_data),
            "metadata": {
                "session_id": self.session_dir.name,
                "generated_at": datetime.now().isoformat()
            }
        }

        logger.info(f"{self.agent_name}: Generated {len(child_shots_data)} child shots")

        return output

    def _extract_child_shots(self, shot_grouping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively extract all child shots from grouping."""
        child_shots = []

        def recurse(grouped_shot, parent_id):
            for child in grouped_shot.get("child_shots", []):
                child_shots.append({
                    "shot_id": child["shot_id"],
                    "parent_shot_id": parent_id
                })
                # Recursively process nested children
                recurse(child, child["shot_id"])

        # Process all parent shots
        for parent in shot_grouping.get("parent_shots", []):
            recurse(parent, parent["shot_id"])

        return child_shots

    def _get_character_physical_descriptions(
        self,
        character_names: List[str],
        scene_breakdown: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract full physical descriptions for characters."""
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

    def _detect_edit_type(
        self,
        shot_id: str,
        parent_id: str,
        shot_details: Dict[str, Any],
        shots_by_id: Dict[str, Any]
    ) -> str:
        """
        Detect the type of edit operation.

        Returns: 'add_character', 'remove_character', 'expression_change', 'camera_change'
        """
        parent_chars = set(shots_by_id.get(parent_id, {}).get("characters", []))
        child_chars = set(shot_details.get("characters", []))

        if len(child_chars) > len(parent_chars):
            return "add_character"
        elif len(child_chars) < len(parent_chars):
            return "remove_character"
        elif parent_chars != child_chars:
            return "character_swap"
        else:
            # Same characters - likely camera or expression change
            # Check first_frame for indicators
            first_frame = shot_details.get("first_frame", "").lower()
            if any(word in first_frame for word in ["close-up", "close up", "zoom", "tighter"]):
                return "camera_change"
            elif any(word in first_frame for word in ["expression", "smile", "frown", "angry", "happy"]):
                return "expression_change"
            else:
                return "camera_change"  # Default assumption

    def _optimize_edit_prompt_with_pro(self, verbose_prompt: str, edit_type: str) -> str:
        """
        Use Gemini Pro 2.5 to optimize the verbose edit prompt following best practices.

        Args:
            verbose_prompt: The filled template prompt
            edit_type: Type of edit detected

        Returns:
            Optimized edit prompt for Flash Image
        """
        from google.genai import types

        # Check if optimization is enabled
        if not self.use_optimizer:
            logger.debug("Prompt optimization disabled in config")
            return verbose_prompt

        # Load optimizer template
        optimizer_template_path = Path("prompts/agent_8_optimizer_prompt.txt")
        if not optimizer_template_path.exists():
            logger.warning("Optimizer template not found, using verbose prompt as-is")
            return verbose_prompt

        with open(optimizer_template_path, 'r', encoding='utf-8') as f:
            optimizer_template = f.read()

        # Create optimization prompt
        optimization_prompt = optimizer_template.format(
            verbose_edit_description=verbose_prompt
        )

        try:
            # Use Pro 2.5 to optimize
            logger.debug(f"Optimizing edit prompt with Pro 2.5 (edit_type: {edit_type})...")
            response = self.client.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[optimization_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Very low temp for precise, consistent edit instructions
                    max_output_tokens=1500
                )
            )

            if response.text:
                optimized = response.text.strip()
                logger.info(f"Edit prompt optimized: {len(verbose_prompt)} → {len(optimized)} chars (type: {edit_type})")
                return optimized
            else:
                logger.warning("Pro optimization returned empty, using verbose prompt")
                return verbose_prompt

        except Exception as e:
            logger.warning(f"Edit prompt optimization failed: {str(e)}, using verbose prompt")
            return verbose_prompt

    def _generate_child_shot_image(
        self,
        shot_id: str,
        parent_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Path,
        grids_by_chars: Dict[tuple, Path],
        shots_by_id: Dict[str, Any]
    ) -> Path:
        """Generate child shot by editing parent shot."""
        from google.genai import types

        # Detect edit type for optimizer
        edit_type = self._detect_edit_type(shot_id, parent_id, shot_details, shots_by_id)
        logger.debug(f"Edit type detected: {edit_type}")

        # Load parent image (PRIMARY INPUT #1)
        parent_image = Image.open(parent_image_path)
        logger.debug(f"Using parent image as input: {parent_image_path.name}")

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location_name = shot_details.get("location", "")
        scene_id = shot_details.get("scene_id", "")
        dialogue = shot_details.get("dialogue", "")

        # Get verbose character descriptions (PHYSICAL TRAITS!)
        character_descriptions = self._get_character_physical_descriptions(
            characters,
            scene_breakdown
        )

        # Find matching character grid (INPUT #2 if applicable)
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        # Format character descriptions into template string
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""
CHARACTER {idx}:
PHYSICAL TRAITS: {char['physical_description']}

When editing the parent shot, identify this character by these exact physical traits. If adding this character to the scene, use the character grid reference (if provided) to ensure their appearance matches precisely.

"""

        # Use template from prompt file
        if not self.prompt_template:
            raise ValueError("Prompt template not loaded")

        verbose_prompt = self.prompt_template.format(
            shot_id=shot_id,
            parent_id=parent_id,
            first_frame=first_frame,
            character_descriptions=char_desc_text,
            location=location_name
        )

        # Optimize edit prompt using Pro 2.5 following best practices
        optimized_prompt = self._optimize_edit_prompt_with_pro(verbose_prompt, edit_type)

        # Prepare contents: [parent image, character grid (if available), optimized edit prompt]
        contents = [parent_image]

        if grid_path and grid_path.exists():
            grid_image = Image.open(grid_path)
            contents.append(grid_image)
            logger.debug(f"Using character grid for editing: {grid_path.name}")

        contents.append(optimized_prompt)
        logger.debug(f"Using optimized edit prompt (type: {edit_type})")

        # Generate edited image
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
        image_filename = f"{shot_id}_child.png"
        image_path = self.child_shots_dir / image_filename

        save_image_with_metadata(
            generated_image,
            image_path,
            metadata={
                "shot_id": shot_id,
                "parent_shot_id": parent_id,
                "characters": characters,
                "location": location_name,
                "generated_at": datetime.now().isoformat(),
                "edit_type": edit_type,
                "grid_used": str(grid_path) if grid_path and grid_path.exists() else None,
                "prompts": {
                    "verbose_prompt": verbose_prompt,
                    "optimized_prompt": optimized_prompt,
                    "optimizer_used": self.use_optimizer
                }
            }
        )

        return image_path
