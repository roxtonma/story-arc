"""
Gemini API Client
Centralized wrapper for Google Gemini API with retry logic and error handling.
Updated for google-genai SDK (unified SDK, GA May 2025).
"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials


class GeminiClient:
    """Wrapper for Google Gemini API with built-in retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 3,
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials_file: Optional[str] = None
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key. If None, reads from GEMINI_API_KEY or GOOGLE_API_KEY env var
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retry attempts
            use_vertex_ai: Whether to use Vertex AI instead of direct API
            vertex_project: GCP project ID (for Vertex AI)
            vertex_location: GCP location/region (for Vertex AI)
            vertex_credentials_file: Path to service account JSON file (for Vertex AI)
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_vertex_ai = use_vertex_ai

        # Initialize client based on authentication mode
        if use_vertex_ai:
            self._init_vertex_ai(vertex_project, vertex_location, vertex_credentials_file)
        else:
            self._init_api_key(api_key)

        logger.info(f"Initialized GeminiClient with model: {model_name}")

    def _init_api_key(self, api_key: Optional[str]) -> None:
        """
        Initialize client for direct API with API key.

        Args:
            api_key: Google API key

        Raises:
            ValueError: If API key is not provided
        """
        # Check both environment variables (GEMINI_API_KEY preferred, GOOGLE_API_KEY fallback)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable "
                "or pass api_key parameter.\n\n"
                "Get your API key at: https://aistudio.google.com/apikey\n"
                "For paid tier access: Enable billing at https://aistudio.google.com/api-keys"
            )

        # Create client with API key
        self.client = genai.Client(api_key=self.api_key)

        logger.info("Using direct Gemini API with API key")
        logger.info("Note: Tier (free/paid) is determined by billing setup in Google AI Studio")

    def _init_vertex_ai(
        self,
        project: Optional[str],
        location: Optional[str],
        credentials_file: Optional[str]
    ) -> None:
        """
        Initialize client for Vertex AI with service account.

        Args:
            project: GCP project ID
            location: GCP location/region
            credentials_file: Path to service account JSON file

        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If credentials file doesn't exist
        """
        # Get project ID from parameter or environment variable
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError(
                "GCP project ID required for Vertex AI.\n"
                "Set GOOGLE_CLOUD_PROJECT environment variable or pass vertex_project parameter.\n"
                "Example: GOOGLE_CLOUD_PROJECT=your-project-id"
            )

        # Get location from parameter or environment variable (default: us-central1)
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Get credentials file path
        creds_file = credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google-credentials.json")
        creds_path = Path(creds_file)

        if not creds_path.exists():
            raise FileNotFoundError(
                f"Service account credentials file not found: {creds_file}\n\n"
                "To create a service account:\n"
                "1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts\n"
                "2. Create service account with 'Vertex AI User' role\n"
                "3. Create and download JSON key\n"
                "4. Save as 'google-credentials.json' in project root\n"
                "5. IMPORTANT: Add to .gitignore to protect credentials"
            )

        # Load credentials with required scope
        logger.info(f"Loading service account credentials from: {creds_file}")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        try:
            credentials = Credentials.from_service_account_file(
                str(creds_path),
                scopes=scopes
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load service account credentials: {str(e)}\n"
                "Ensure the JSON file is valid and properly formatted."
            )

        # Initialize client with Vertex AI
        try:
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                credentials=credentials
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Vertex AI client: {str(e)}\n"
                "Ensure Vertex AI API is enabled in your GCP project."
            )

        logger.info(f"Using Vertex AI (project: {self.project}, location: {self.location})")
        logger.info("Note: Ensure service account has 'Vertex AI User' role")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
        retry_on_error: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API with retry logic.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_output_tokens: Maximum tokens in response
            system_instruction: Optional system instruction
            retry_on_error: Whether to retry on API errors
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Raises:
            Exception: If all retry attempts fail
        """
        # Build generation config (new SDK pattern - system instruction in config)
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,  # Now part of config instead of model
            **kwargs
        )

        last_error = None
        retries = self.max_retries if retry_on_error else 1

        for attempt in range(retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries} for Gemini API call")

                # New SDK pattern - use client.models.generate_content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config
                )

                # Check if response has text
                if not response.text:
                    raise ValueError("Empty response from Gemini API")

                logger.info(f"Successfully generated response (length: {len(response.text)})")
                return response.text

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

                # Check for common billing/quota errors
                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning(
                        "Rate limit reached. If on free tier, consider enabling billing at: "
                        "https://aistudio.google.com/api-keys"
                    )

                if attempt < retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {retries} attempts failed")

        raise Exception(f"Failed to generate content after {retries} attempts: {last_error}")

    def generate_with_feedback(
        self,
        prompt: str,
        error_feedback: str,
        **kwargs
    ) -> str:
        """
        Generate content with error feedback from previous attempt.

        Args:
            prompt: Original prompt
            error_feedback: Description of what went wrong
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        feedback_prompt = f"""
{prompt}

IMPORTANT: The previous attempt failed with the following issue:
{error_feedback}

Please correct this issue and try again.
"""
        return self.generate(feedback_prompt, **kwargs)

    def validate_api_key(self) -> bool:
        """
        Validate that the API key works.

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Say 'API key valid' if you can read this.",
                config=types.GenerateContentConfig(max_output_tokens=10)
            )
            is_valid = bool(response.text)

            if is_valid:
                logger.info("✓ API key is valid")
                logger.info(
                    "Tier (free/paid) is determined by billing setup in Google AI Studio\n"
                    "To enable paid tier: https://aistudio.google.com/api-keys"
                )

            return is_valid
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            logger.error("Get your API key at: https://aistudio.google.com/apikey")
            return False

    def get_tier_info(self) -> Dict[str, Any]:
        """
        Get information about API tier configuration.

        Note: Tier (free vs. paid) is determined by billing setup in Google AI Studio,
        not by API configuration. This method provides guidance on how to configure tier.

        Returns:
            Dictionary with tier information and setup instructions
        """
        tier_info = {
            "tier_configuration": "Account-based (not code-based)",
            "free_tier_limits": {
                "rate_limits": "Lower (15 RPM for Gemini 1.5 Pro)",
                "features": "Limited context caching, no batch API",
                "data_usage": "May be used to improve Google products"
            },
            "paid_tier_benefits": {
                "rate_limits": "Higher (up to 1000+ RPM depending on model)",
                "features": "Full context caching, batch API access, priority support",
                "data_usage": "Not used to improve Google products"
            },
            "how_to_enable_paid_tier": [
                "1. Visit: https://aistudio.google.com/api-keys",
                "2. Click 'Set up Billing' or 'Upgrade' button",
                "3. Link a Google Cloud billing account",
                "4. Same API key automatically uses paid tier after billing is enabled"
            ],
            "billing_setup_url": "https://aistudio.google.com/api-keys",
            "documentation_url": "https://ai.google.dev/gemini-api/docs/pricing"
        }

        logger.info("API Tier Information:")
        logger.info(f"  Configuration: {tier_info['tier_configuration']}")
        logger.info(f"  Billing Setup: {tier_info['billing_setup_url']}")

        return tier_info
