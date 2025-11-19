"""
Agent 3: Shot Breakdown
Converts scenes into individual shots with strict JSON format.
Each shot has: shot_description, first_frame (with ALL elements), and animation.
"""

import json
from typing import Any, Dict
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ShotBreakdown


class ShotBreakdownAgent(BaseAgent):
    """Agent for breaking down scenes into individual shots."""

    def __init__(self, gemini_client, config):
        """Initialize Shot Breakdown Agent."""
        super().__init__(gemini_client, config, "agent_3")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input scene breakdown.

        Args:
            input_data: Scene breakdown dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary (scene breakdown)")

        if "scenes" not in input_data:
            raise ValueError("Input must contain 'scenes' key")

        if not isinstance(input_data["scenes"], list):
            raise ValueError("'scenes' must be a list")

        if len(input_data["scenes"]) == 0:
            raise ValueError("Scene breakdown must contain at least one scene")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output shot breakdown.

        Args:
            output_data: Shot breakdown dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        # Validate using Pydantic model
        try:
            shot_breakdown = ShotBreakdown(**output_data)

            # Additional validation
            if shot_breakdown.total_shots == 0:
                raise ValueError("No shots found in breakdown")

            # Validate each shot
            for shot in shot_breakdown.shots:
                # Check description length
                if len(shot.shot_description) < 30:
                    raise ValueError(
                        f"Shot description for {shot.shot_id} is too short "
                        f"(minimum 30 characters)"
                    )

                # Check first frame verbosity (validated in Shot model, but double-check)
                if len(shot.first_frame) < 40:
                    raise ValueError(
                        f"First frame description for {shot.shot_id} must be more verbose "
                        f"(minimum 50 characters)"
                    )

                # Check animation description
                if len(shot.animation) < 20:
                    raise ValueError(
                        f"Animation description for {shot.shot_id} is too short "
                        f"(minimum 20 characters)"
                    )

                # Ensure shot_id and scene_id are present
                if not shot.shot_id or not shot.scene_id:
                    raise ValueError(
                        f"Shot must have both shot_id and scene_id"
                    )

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Shot breakdown validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Break down scenes into shots.

        Args:
            input_data: Scene breakdown dictionary

        Returns:
            Shot breakdown dictionary
        """
        logger.info(f"{self.agent_name}: Breaking down scenes into shots...")

        # Convert scene breakdown to JSON string for prompt
        scene_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # Format prompt with scene breakdown
        prompt = self.format_prompt(scene_breakdown_json)

        # Generate shot breakdown using Gemini
        response = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            shot_breakdown = self._extract_json(response)

            # Auto-calculate total_shots from actual array length (source of truth)
            shot_breakdown["total_shots"] = len(shot_breakdown.get("shots", []))

            logger.info(
                f"{self.agent_name}: Generated breakdown with "
                f"{shot_breakdown.get('total_shots', 0)} shots"
            )

            return shot_breakdown

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse shot breakdown JSON: {str(e)}")

    def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str
    ) -> Dict[str, Any]:
        """
        Generate shot breakdown with error feedback.

        Args:
            input_data: Scene breakdown dictionary
            error_feedback: Feedback about what went wrong

        Returns:
            Regenerated shot breakdown
        """
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        scene_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)
        prompt = self.format_prompt(scene_breakdown_json)

        # Generate with feedback
        response = self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            shot_breakdown = self._extract_json(response)
            # Auto-calculate total_shots from actual array length (source of truth)
            shot_breakdown["total_shots"] = len(shot_breakdown.get("shots", []))
            return shot_breakdown
        except Exception as e:
            raise ValueError(f"Failed to parse shot breakdown JSON: {str(e)}")

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
                return {"shots": data, "total_shots": len(data), "metadata": {}}
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")
