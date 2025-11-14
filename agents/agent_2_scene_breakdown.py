"""
Agent 2: Scene Breakdown
Breaks down screenplay into scenes with verbose location/character descriptions and subscenes.
"""

import json
from typing import Any, Dict
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import SceneBreakdown


class SceneBreakdownAgent(BaseAgent):
    """Agent for breaking down screenplay into detailed scenes."""

    def __init__(self, gemini_client, config):
        """Initialize Scene Breakdown Agent."""
        super().__init__(gemini_client, config, "agent_2")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input screenplay.

        Args:
            input_data: Screenplay text

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string (screenplay text)")

        if len(input_data.strip()) < 100:
            raise ValueError(
                "Screenplay is too short. Please provide a complete screenplay."
            )

        # Check for basic screenplay formatting
        screenplay = input_data.upper()
        has_scene_headings = any(
            marker in screenplay
            for marker in ["INT.", "EXT.", "INT/EXT"]
        )

        if not has_scene_headings:
            raise ValueError(
                "Input must be a properly formatted screenplay with scene headings"
            )

        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output scene breakdown.

        Args:
            output_data: Scene breakdown dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        # Validate using Pydantic model
        try:
            scene_breakdown = SceneBreakdown(**output_data)

            # Additional validation
            if scene_breakdown.total_scenes == 0:
                raise ValueError("No scenes found in breakdown")

            # Check for verbose descriptions
            for scene in scene_breakdown.scenes:
                if len(scene.location.description) < 50:
                    raise ValueError(
                        f"Location description for {scene.scene_id} is not verbose enough "
                        f"(minimum 50 characters, got {len(scene.location.description)})"
                    )

                for char in scene.characters:
                    if len(char.description) < 50:
                        raise ValueError(
                            f"Character description for {char.name} in {scene.scene_id} "
                            f"is not verbose enough (minimum 50 characters, "
                            f"got {len(char.description)})"
                        )

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Scene breakdown validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Break down screenplay into scenes.

        Args:
            input_data: Screenplay text

        Returns:
            Scene breakdown dictionary
        """
        logger.info(f"{self.agent_name}: Breaking down screenplay into scenes...")

        # Format prompt with screenplay
        prompt = self.format_prompt(input_data)

        # Generate scene breakdown using Gemini
        response = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            scene_breakdown = self._extract_json(response)

            # Auto-calculate total_scenes from actual array length (source of truth)
            scene_breakdown["total_scenes"] = len(scene_breakdown.get("scenes", []))

            logger.info(
                f"{self.agent_name}: Generated breakdown with "
                f"{scene_breakdown.get('total_scenes', 0)} scenes"
            )

            return scene_breakdown

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse scene breakdown JSON: {str(e)}")

    def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str
    ) -> Dict[str, Any]:
        """
        Generate scene breakdown with error feedback.

        Args:
            input_data: Screenplay text
            error_feedback: Feedback about what went wrong

        Returns:
            Regenerated scene breakdown
        """
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        prompt = self.format_prompt(input_data)

        # Generate with feedback
        response = self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            scene_breakdown = self._extract_json(response)
            # Auto-calculate total_scenes from actual array length (source of truth)
            scene_breakdown["total_scenes"] = len(scene_breakdown.get("scenes", []))
            return scene_breakdown
        except Exception as e:
            raise ValueError(f"Failed to parse scene breakdown JSON: {str(e)}")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from text response.

        Args:
            text: Response text that may contain JSON

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted
        """
        # Try to parse entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                # Wrap array in expected structure
                return {"scenes": data, "total_scenes": len(data), "metadata": {}}
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")
