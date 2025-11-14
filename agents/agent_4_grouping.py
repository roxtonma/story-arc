"""
Agent 4: Shot Grouping
Groups shots into parent-child relationships for efficient image generation.
Supports cross-scene grouping and multi-level hierarchies.
"""

import json
from typing import Any, Dict
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ShotGrouping


class ShotGroupingAgent(BaseAgent):
    """Agent for grouping shots into parent-child hierarchies."""

    def __init__(self, gemini_client, config):
        """Initialize Shot Grouping Agent."""
        super().__init__(gemini_client, config, "agent_4")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input shot breakdown.

        Args:
            input_data: Shot breakdown dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary (shot breakdown)")

        if "shots" not in input_data:
            raise ValueError("Input must contain 'shots' key")

        if not isinstance(input_data["shots"], list):
            raise ValueError("'shots' must be a list")

        if len(input_data["shots"]) == 0:
            raise ValueError("Shot breakdown must contain at least one shot")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output shot grouping.

        Args:
            output_data: Shot grouping dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        # Validate using Pydantic model
        try:
            shot_grouping = ShotGrouping(**output_data)

            # Additional validation
            if shot_grouping.total_parent_shots == 0:
                raise ValueError("No parent shots found in grouping")

            # Validate hierarchy structure
            self._validate_hierarchy(shot_grouping.parent_shots)

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Shot grouping validation failed: {str(e)}")

    def _validate_hierarchy(self, grouped_shots, parent_id=None):
        """
        Recursively validate shot hierarchy.

        Args:
            grouped_shots: List of GroupedShot objects
            parent_id: Parent shot ID (for child validation)

        Raises:
            ValueError: If hierarchy is invalid
        """
        for shot in grouped_shots:
            # Validate parent_shot_id consistency
            if parent_id and shot.parent_shot_id != parent_id:
                raise ValueError(
                    f"Shot {shot.shot_id} has incorrect parent_shot_id: "
                    f"expected {parent_id}, got {shot.parent_shot_id}"
                )

            # Validate grouping_reason presence
            if not shot.grouping_reason:
                raise ValueError(
                    f"Shot {shot.shot_id} must have a grouping_reason"
                )

            # Recursively validate children (multi-level hierarchies allowed)
            if shot.child_shots:
                self._validate_hierarchy(shot.child_shots, shot.shot_id)

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Group shots into parent-child relationships.

        Args:
            input_data: Shot breakdown dictionary

        Returns:
            Shot grouping dictionary
        """
        logger.info(f"{self.agent_name}: Grouping shots into parent-child relationships...")

        # Convert shot breakdown to JSON string for prompt
        shot_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # Format prompt with shot breakdown
        prompt = self.format_prompt(shot_breakdown_json)

        # Generate shot grouping using Gemini
        response = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            shot_grouping = self._extract_json(response)

            # Auto-calculate counts from actual data (source of truth)
            shot_grouping["total_parent_shots"] = len(shot_grouping.get("parent_shots", []))
            shot_grouping["total_child_shots"] = self._count_children(shot_grouping.get("parent_shots", []))

            logger.info(
                f"{self.agent_name}: Generated grouping with "
                f"{shot_grouping.get('total_parent_shots', 0)} parent shots and "
                f"{shot_grouping.get('total_child_shots', 0)} child shots"
            )

            return shot_grouping

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse shot grouping JSON: {str(e)}")

    def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str
    ) -> Dict[str, Any]:
        """
        Generate shot grouping with error feedback.

        Args:
            input_data: Shot breakdown dictionary
            error_feedback: Feedback about what went wrong

        Returns:
            Regenerated shot grouping
        """
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        shot_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)
        prompt = self.format_prompt(shot_breakdown_json)

        # Generate with feedback
        response = self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        # Parse JSON response
        try:
            shot_grouping = self._extract_json(response)
            # Auto-calculate counts from actual data (source of truth)
            shot_grouping["total_parent_shots"] = len(shot_grouping.get("parent_shots", []))
            shot_grouping["total_child_shots"] = self._count_children(shot_grouping.get("parent_shots", []))
            return shot_grouping
        except Exception as e:
            raise ValueError(f"Failed to parse shot grouping JSON: {str(e)}")

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

        # Try to find JSON array (wrap it)
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                # Wrap array in expected structure
                return {
                    "parent_shots": data,
                    "total_parent_shots": len(data),
                    "total_child_shots": self._count_children(data),
                    "grouping_strategy": "Location and character-based grouping",
                    "metadata": {}
                }
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")

    def _count_children(self, parent_shots):
        """
        Recursively count all child shots.

        Args:
            parent_shots: List of parent shot dictionaries

        Returns:
            Total count of child shots
        """
        count = 0
        for shot in parent_shots:
            if "child_shots" in shot and shot["child_shots"]:
                count += len(shot["child_shots"])
                count += self._count_children(shot["child_shots"])
        return count
