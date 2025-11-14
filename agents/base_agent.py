"""
Base Agent
Abstract base class for all agents in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
from loguru import logger

from core.gemini_client import GeminiClient


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Dict[str, Any],
        agent_name: str
    ):
        """
        Initialize base agent.

        Args:
            gemini_client: Initialized Gemini client
            config: Agent configuration dictionary
            agent_name: Name of the agent (e.g., 'agent_1')
        """
        self.client = gemini_client
        self.config = config
        self.agent_name = agent_name

        # Extract configuration
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 8192)
        self.prompt_file = config.get("prompt_file")

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        logger.info(f"Initialized {self.agent_name}: {config.get('name', 'Unknown')}")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return output.

        Args:
            input_data: Input data for this agent

        Returns:
            Processed output data
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If input is invalid with description
        """
        pass

    @abstractmethod
    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output data after processing.

        Args:
            output_data: Output data to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If output is invalid with description
        """
        pass

    def _load_prompt_template(self) -> Optional[str]:
        """
        Load prompt template from file.

        Returns:
            Prompt template string or None if file not found
        """
        if not self.prompt_file:
            logger.warning(f"{self.agent_name}: No prompt file specified")
            return None

        prompt_path = Path(self.prompt_file)

        if not prompt_path.exists():
            logger.warning(f"{self.agent_name}: Prompt file not found: {self.prompt_file}")
            return None

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()

            logger.debug(f"{self.agent_name}: Loaded prompt template from {self.prompt_file}")
            return template

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to load prompt template: {str(e)}")
            return None

    def format_prompt(self, input_data: Any, **kwargs) -> str:
        """
        Format prompt template with input data.

        Args:
            input_data: Input data to insert into template
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        if not self.prompt_template:
            raise ValueError(f"{self.agent_name}: No prompt template available")

        try:
            # Create template variables
            template_vars = {
                "input": input_data,
                **kwargs
            }

            # Format template
            prompt = self.prompt_template.format(**template_vars)
            return prompt

        except KeyError as e:
            raise ValueError(f"{self.agent_name}: Missing template variable: {str(e)}")

    def execute(self, input_data: Any, retry_count: int = 0) -> Any:
        """
        Execute the agent with input validation, processing, and output validation.

        Args:
            input_data: Input data for processing
            retry_count: Current retry attempt number

        Returns:
            Validated output data

        Raises:
            ValueError: If validation fails
            Exception: If processing fails
        """
        logger.info(f"{self.agent_name}: Starting execution (attempt {retry_count + 1})")

        # Validate input
        try:
            self.validate_input(input_data)
            logger.debug(f"{self.agent_name}: Input validation passed")
        except Exception as e:
            logger.error(f"{self.agent_name}: Input validation failed: {str(e)}")
            raise

        # Process
        try:
            output_data = self.process(input_data)
            logger.debug(f"{self.agent_name}: Processing completed")
        except Exception as e:
            logger.error(f"{self.agent_name}: Processing failed: {str(e)}")
            raise

        # Validate output
        try:
            self.validate_output(output_data)
            logger.debug(f"{self.agent_name}: Output validation passed")
        except Exception as e:
            logger.error(f"{self.agent_name}: Output validation failed: {str(e)}")
            raise

        logger.info(f"{self.agent_name}: Execution completed successfully")
        return output_data

    def execute_with_retry(
        self,
        input_data: Any,
        max_retries: int = 3,
        error_feedback: Optional[str] = None
    ) -> Any:
        """
        Execute agent with automatic retry on failure.

        Args:
            input_data: Input data for processing
            max_retries: Maximum number of retry attempts
            error_feedback: Optional error feedback from previous attempt

        Returns:
            Validated output data

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"{self.agent_name}: Attempt {attempt + 1}/{max_retries}")

                # Use error feedback if provided and not first attempt
                if error_feedback and attempt > 0:
                    logger.info(f"{self.agent_name}: Using error feedback for retry")
                    # This will be handled differently by each agent
                    pass

                output_data = self.execute(input_data, retry_count=attempt)
                return output_data

            except Exception as e:
                last_error = e
                logger.warning(f"{self.agent_name}: Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    logger.info(f"{self.agent_name}: Retrying...")
                    # Prepare error feedback for next attempt
                    error_feedback = str(e)

        # All attempts failed
        logger.error(f"{self.agent_name}: All {max_retries} attempts failed")
        raise Exception(
            f"{self.agent_name} failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary with agent information
        """
        return {
            "agent_name": self.agent_name,
            "display_name": self.config.get("name", "Unknown"),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "prompt_file": self.prompt_file,
            "enabled": self.config.get("enabled", True)
        }
