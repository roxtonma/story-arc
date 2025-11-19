"""
Agent 5: Character Creator
Generates consistent character images and combination grids for shot reference.
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import CharacterCreationOutput
from core.image_utils import (
    extract_character_combinations,
    create_character_grid,
    save_image_with_metadata
)
from utils.fal_helper import (
    generate_with_fal_text_to_image,
    is_fal_available
)


class CharacterCreatorAgent(BaseAgent):
    """Agent for generating character images and combination grids."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Character Creator Agent."""
        super().__init__(gemini_client, config, "agent_5")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.characters_dir = self.assets_dir / "characters"
        self.grids_dir = self.assets_dir / "grids"

        # Create directories
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.grids_dir.mkdir(parents=True, exist_ok=True)

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data (Agent 2 scene breakdown + Agent 3 shot breakdown).

        Args:
            input_data: Dict with 'scene_breakdown' and 'shot_breakdown' keys

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        if "scene_breakdown" not in input_data:
            raise ValueError("Input must contain 'scene_breakdown' from Agent 2")

        if "shot_breakdown" not in input_data:
            raise ValueError("Input must contain 'shot_breakdown' from Agent 3")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output character creation data.

        Args:
            output_data: Character creation dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        # Validate using Pydantic model
        try:
            char_output = CharacterCreationOutput(**output_data)

            if char_output.total_characters == 0:
                raise ValueError("No characters generated")

            if len(char_output.characters) != char_output.total_characters:
                # Auto-fix count mismatch
                output_data["total_characters"] = len(char_output.characters)

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Character creation validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Generate character images and combination grids.

        Args:
            input_data: Dict with scene_breakdown and shot_breakdown

        Returns:
            Character creation output dictionary
        """
        logger.info(f"{self.agent_name}: Generating character images and grids...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]

        # Step 1: Extract unique characters from Agent 2 and Agent 3
        characters = self._extract_characters(scene_breakdown, shot_breakdown)
        logger.info(f"Found {len(characters)} unique characters")

        # Step 2: Generate character images
        character_data = []
        character_images = {}

        for char_name, char_desc in characters.items():
            try:
                logger.info(f"Generating image for character: {char_name}")

                # Generate character image
                char_image = self._generate_character_image(char_name, char_desc)

                # Save character image
                char_filename = f"char_{self._slugify(char_name)}.png"
                char_path = self.characters_dir / char_filename

                save_image_with_metadata(
                    char_image,
                    char_path,
                    metadata={
                        "character_name": char_name,
                        "description": char_desc,
                        "generated_at": datetime.now().isoformat()
                    }
                )

                # Store data
                character_data.append({
                    "name": char_name,
                    "description": char_desc,
                    "image_path": f"assets/characters/{char_filename}",
                    "generation_timestamp": datetime.now().isoformat()
                })

                character_images[char_name] = char_image

                logger.info(f"✓ Generated character: {char_name}")

            except Exception as e:
                logger.error(f"Failed to generate character {char_name}: {str(e)}")
                raise

        # Step 3: Analyze shots to find needed character combinations
        logger.debug(f"Shot breakdown has {len(shot_breakdown.get('shots', []))} shots")
        combinations = extract_character_combinations(shot_breakdown)
        logger.info(f"Found {len(combinations)} unique character combinations: {combinations}")
        logger.debug(f"Available character images: {list(character_images.keys())}")

        # Step 4: Generate character grids
        grid_data = []

        for combo in combinations:
            try:
                combo_str = "_".join(sorted(combo))
                logger.info(f"Creating grid for: {combo_str}")
                logger.debug(f"Combo details: {combo}, type: {type(combo)}")

                # Get character images for this combination
                combo_images = [character_images[name] for name in combo if name in character_images]
                logger.debug(f"Retrieved {len(combo_images)} images for {len(combo)} characters in combo")

                if not combo_images:
                    logger.warning(f"No images found for combination: {combo}")
                    logger.warning(f"Available keys: {list(character_images.keys())}, Requested: {list(combo)}")
                    continue

                # Create grid
                logger.debug(f"Calling create_character_grid() with {len(combo_images)} images")
                grid_image = create_character_grid(
                    combo_images,
                    list(combo),
                    output_size=(1920, 1080)  # 16:9 aspect ratio
                )
                logger.debug(f"Grid created successfully, size: {grid_image.size}")

                # Save grid
                grid_filename = f"grid_{combo_str}.png"
                grid_path = self.grids_dir / grid_filename

                save_image_with_metadata(
                    grid_image,
                    grid_path,
                    metadata={
                        "characters": list(combo),
                        "generated_at": datetime.now().isoformat()
                    }
                )

                # Store data
                grid_data.append({
                    "grid_id": combo_str,
                    "characters": list(combo),
                    "grid_path": f"assets/grids/{grid_filename}",
                    "generation_timestamp": datetime.now().isoformat()
                })

                logger.info(f"✓ Created grid: {combo_str}")

            except Exception as e:
                logger.error(f"GRID CREATION FAILED for {combo}: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.exception("Full traceback:")
                # Continue with other grids (soft failure for grids)
                continue

        # Prepare output
        output = {
            "characters": character_data,
            "character_grids": grid_data,
            "total_characters": len(character_data),
            "total_grids": len(grid_data),
            "metadata": {
                "session_id": self.session_dir.name,
                "generated_at": datetime.now().isoformat()
            }
        }

        logger.info(
            f"{self.agent_name}: Generated {len(character_data)} characters "
            f"and {len(grid_data)} grids"
        )

        return output

    def _extract_characters(self, scene_breakdown: Dict[str, Any], shot_breakdown: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract unique characters and their descriptions from both scene and shot breakdowns.
        Checks main scenes, subscenes (CHARACTER_ADDED events), and shot data.

        Args:
            scene_breakdown: Agent 2 output
            shot_breakdown: Agent 3 output

        Returns:
            Dict mapping character name to visual description
        """
        characters = {}

        for scene in scene_breakdown.get("scenes", []):
            # Extract from main scene characters
            for char in scene.get("characters", []):
                char_name = char.get("name")
                char_desc = char.get("description")

                if char_name and char_desc and char_name not in characters:
                    characters[char_name] = char_desc

            # Extract from subscenes (CHARACTER_ADDED events)
            for subscene in scene.get("subscenes", []):
                if subscene.get("event") == "CHARACTER_ADDED":
                    char_added = subscene.get("character_added", {})
                    char_name = char_added.get("name")
                    char_desc = char_added.get("description")

                    if char_name and char_desc and char_name not in characters:
                        characters[char_name] = char_desc
                        logger.info(f"Found character '{char_name}' in subscene CHARACTER_ADDED event")

        # Cross-reference with shot_breakdown to catch any missed characters
        shot_characters = set()
        for shot in shot_breakdown.get("shots", []):
            shot_characters.update(shot.get("characters", []))

        # For characters in shots but not in our dict, create placeholder
        for char_name in shot_characters:
            if char_name not in characters:
                logger.warning(
                    f"Character '{char_name}' found in shots but not in scene breakdown. "
                    "Creating placeholder description."
                )
                characters[char_name] = (
                    f"A character named {char_name}. "
                    "(No detailed description available from scene breakdown - "
                    "this character may be a minor role or background character)"
                )

        return characters

    def _generate_character_image(self, char_name: str, char_desc: str) -> Image.Image:
        """
        Generate character image using configured image provider (Gemini or fal).

        Args:
            char_name: Character name
            char_desc: Character description

        Returns:
            PIL Image object
        """
        # Format prompt for character generation
        prompt = f"""
Generate a high-quality character portrait for:

Character Name: {char_name}

Physical Description: {char_desc}

Style Requirements:
- Professional character design
- Clean, neutral background (white or soft gradient)
- Studio lighting, well-lit face
- Cinematic quality
- Front-facing portrait
- Consistent character design suitable for video generation reference

Generate a clear, detailed portrait of this character.
"""

        # Check image provider from config
        image_provider = self.config.get("image_provider", "gemini").lower()

        if image_provider == "fal":
            # Use fal for image generation
            logger.info(f"Using fal for character image generation: {char_name}")

            if not is_fal_available():
                logger.warning("fal_client not available, falling back to Gemini")
                image_provider = "gemini"
            else:
                try:
                    fal_model = self.config.get(
                        "fal_text_to_image_model",
                        "fal-ai/bytedance/seedream/v4/text-to-image"
                    )

                    # Use 1024x1024 for character portraits (1:1 aspect ratio)
                    pil_image, seed = generate_with_fal_text_to_image(
                        prompt=prompt,
                        model=fal_model,
                        width=1024,
                        height=1024,
                        num_images=1,
                        enable_safety_checker=True,
                        enhance_prompt_mode="standard"
                    )

                    logger.info(f"Successfully generated character image with fal (seed: {seed})")
                    return pil_image

                except Exception as e:
                    logger.error(f"Failed to generate with fal: {e}, falling back to Gemini")
                    image_provider = "gemini"

        # Use Gemini for image generation (default or fallback)
        if image_provider == "gemini":
            from google.genai import types

            logger.info(f"Using Gemini for character image generation: {char_name}")

            # Generate image
            response = self.client.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",  # 1024x1024 square
                    ),
                ),
            )

            # Extract generated image and convert to PIL Image
            for part in response.parts:
                if part.inline_data is not None:
                    # Get Gemini SDK Image wrapper
                    gemini_image = part.as_image()
                    # Convert to PIL Image
                    pil_image = Image.open(BytesIO(gemini_image.image_bytes))
                    return pil_image

        raise ValueError(f"No image generated for character: {char_name}")

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text if text else "unnamed"
