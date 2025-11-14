"""
Agent 1: Screenplay Generator
Converts input (logline/story/script) into a dialogue-driven screenplay with no narration.
"""

from typing import Any
from loguru import logger

from agents.base_agent import BaseAgent


class ScreenplayAgent(BaseAgent):
    """Agent for generating dialogue-driven screenplays."""

    def __init__(self, gemini_client, config):
        """Initialize Screenplay Agent."""
        super().__init__(gemini_client, config, "agent_1")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input story/logline/script

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string")

        if len(input_data.strip()) < 10:
            raise ValueError("Input is too short. Please provide at least 10 characters.")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output screenplay.

        Args:
            output_data: Generated screenplay

        Returns:
            True if valid

        Raises:
            ValueError: If output is invalid
        """
        if not isinstance(output_data, str):
            raise ValueError("Output must be a string")

        if len(output_data.strip()) < 100:
            raise ValueError("Generated screenplay is too short")

        # Check for basic screenplay formatting
        screenplay = output_data.upper()

        # Should have scene headings
        has_scene_headings = any(
            marker in screenplay
            for marker in ["INT.", "EXT.", "INT/EXT"]
        )

        if not has_scene_headings:
            raise ValueError(
                "Screenplay must contain proper scene headings (INT./EXT.)"
            )

        # Should have dialogue (character names in caps followed by dialogue)
        lines = output_data.split('\n')
        has_dialogue = False

        for i, line in enumerate(lines):
            # Character names are typically all caps and centered/indented
            if line.strip() and line.strip().isupper() and len(line.strip()) > 2:
                # Check if next non-empty line could be dialogue
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip() and not lines[j].strip().isupper():
                        has_dialogue = True
                        break

        if not has_dialogue:
            raise ValueError(
                "Screenplay must be dialogue-driven with character names and dialogue"
            )

        logger.debug(f"{self.agent_name}: Output validation passed")
        return True

    def process(self, input_data: Any) -> str:
        """
        Generate screenplay from input.

        Args:
            input_data: Input story/logline/script

        Returns:
            Generated dialogue-driven screenplay
        """
        logger.info(f"{self.agent_name}: Generating screenplay...")

        # Format prompt with input
        prompt = self.format_prompt(input_data)

        # Generate screenplay using Gemini
        screenplay = self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        logger.info(
            f"{self.agent_name}: Generated screenplay "
            f"({len(screenplay)} characters, {len(screenplay.split())} words)"
        )

        return screenplay

    def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str
    ) -> str:
        """
        Generate screenplay with error feedback.

        Args:
            input_data: Input story/logline/script
            error_feedback: Feedback about what went wrong

        Returns:
            Regenerated screenplay
        """
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        # Format prompt with input and feedback
        prompt = self.format_prompt(input_data)

        # Generate with feedback
        screenplay = self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )

        return screenplay
