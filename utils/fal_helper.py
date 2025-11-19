"""
fal_client integration helper for image generation and editing.

This module provides utility functions to interact with fal.ai's image generation
services, supporting both text-to-image and image-to-image editing.

Environment Variables Required:
    FAL_KEY: Your fal.ai API key (get from https://fal.ai/dashboard/keys)
"""

import logging
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from io import BytesIO
import requests
from PIL import Image

try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False
    logging.warning("fal_client not installed. Install with: pip install fal-client")


logger = logging.getLogger(__name__)


def _check_fal_authentication() -> bool:
    """
    Check if FAL_KEY is configured in environment variables.

    Returns:
        True if authenticated, False otherwise
    """
    fal_key = os.getenv("FAL_KEY")

    if not fal_key or fal_key == "your_fal_api_key_here":
        logger.error(
            "FAL_KEY not configured. Please set your fal.ai API key in .env file. "
            "Get your key from: https://fal.ai/dashboard/keys"
        )
        return False

    return True


def _on_queue_update(update):
    """
    Callback function for fal_client queue updates.
    Logs progress messages during image generation.
    """
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            logger.info(f"fal progress: {log['message']}")


def _download_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a URL and return as PIL Image.

    Args:
        url: URL of the image to download

    Returns:
        PIL Image object

    Raises:
        requests.RequestException: If download fails
        PIL.UnidentifiedImageError: If image format is invalid
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to open image from {url}: {e}")
        raise


def _retry_with_backoff(func, max_retries=3, initial_delay=1.0):
    """
    Retry a function with exponential backoff on failure.

    Args:
        func: Function to retry (should be a callable with no args)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry

    Returns:
        Result from the function

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")

    raise last_exception


def _upload_image_to_fal(image_path: Path) -> str:
    """
    Upload a local image to fal for use in image-to-image operations.
    Includes retry logic for network errors.

    Args:
        image_path: Path to the local image file

    Returns:
        URL of the uploaded image

    Raises:
        Exception: If upload fails after retries
    """
    def _upload():
        # fal_client.upload_file expects a file path string
        url = fal_client.upload_file(str(image_path))
        logger.info(f"Uploaded {image_path.name} to fal")
        return url

    try:
        return _retry_with_backoff(_upload, max_retries=3, initial_delay=2.0)
    except Exception as e:
        logger.error(f"Failed to upload image {image_path} after retries: {e}")
        raise


def generate_with_fal_text_to_image(
    prompt: str,
    model: str = "fal-ai/bytedance/seedream/v4/text-to-image",
    width: int = 4096,
    height: int = 4096,
    num_images: int = 1,
    enable_safety_checker: bool = True,
    enhance_prompt_mode: str = "standard"
) -> Tuple[Image.Image, int]:
    """
    Generate an image from text using fal.ai's text-to-image models.

    Args:
        prompt: Text description of the image to generate
        model: fal model ID to use for generation
        width: Output image width in pixels
        height: Output image height in pixels
        num_images: Number of images to generate (default 1)
        enable_safety_checker: Whether to enable safety checks
        enhance_prompt_mode: Prompt enhancement mode ("standard", "creative", etc.)

    Returns:
        Tuple of (PIL Image object, seed used for generation)

    Raises:
        ImportError: If fal_client is not installed
        Exception: If image generation or download fails or FAL_KEY not configured
    """
    if not FAL_AVAILABLE:
        raise ImportError("fal_client not installed. Install with: pip install fal-client")

    # Check authentication
    if not _check_fal_authentication():
        raise Exception("FAL_KEY not configured. Please set your fal.ai API key in .env file.")

    logger.info(f"Generating image with fal model: {model}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Size: {width}x{height}")

    def _generate():
        result = fal_client.subscribe(
            model,
            arguments={
                "prompt": prompt,
                "image_size": {
                    "height": height,
                    "width": width
                },
                "num_images": num_images,
                "max_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "enhance_prompt_mode": enhance_prompt_mode
            },
            with_logs=True,
            on_queue_update=_on_queue_update,
        )

        logger.info(f"fal generation complete. Seed: {result.get('seed', 'N/A')}")

        # Extract image URL from result
        if not result.get("images") or len(result["images"]) == 0:
            raise Exception("No images returned from fal")

        image_url = result["images"][0]["url"]
        seed = result.get("seed", 0)

        # Download image
        logger.info(f"Downloading image from: {image_url}")
        image = _download_image_from_url(image_url)

        logger.info(f"Successfully generated {width}x{height} image")
        return image, seed

    try:
        return _retry_with_backoff(_generate, max_retries=2, initial_delay=3.0)
    except Exception as e:
        logger.error(f"Failed to generate image with fal after retries: {e}")
        raise


def generate_with_fal_edit(
    prompt: str,
    image_paths: List[Path],
    model: str = "fal-ai/bytedance/seedream/v4/edit",
    width: int = 3840,
    height: int = 2160,
    num_images: int = 1,
    enable_safety_checker: bool = True,
    enhance_prompt_mode: str = "standard"
) -> Tuple[Image.Image, int]:
    """
    Edit/transform existing images using fal.ai's image editing models.

    This is used for image-to-image operations where you want to modify
    or transform existing images based on a text prompt.

    Args:
        prompt: Text description of the edits to apply
        image_paths: List of paths to input images (local files)
        model: fal model ID to use for editing
        width: Output image width in pixels
        height: Output image height in pixels
        num_images: Number of images to generate (default 1)
        enable_safety_checker: Whether to enable safety checks
        enhance_prompt_mode: Prompt enhancement mode ("standard", "creative", etc.)

    Returns:
        Tuple of (PIL Image object, seed used for generation)

    Raises:
        ImportError: If fal_client is not installed
        Exception: If image upload, generation, or download fails or FAL_KEY not configured
    """
    if not FAL_AVAILABLE:
        raise ImportError("fal_client not installed. Install with: pip install fal-client")

    # Check authentication
    if not _check_fal_authentication():
        raise Exception("FAL_KEY not configured. Please set your fal.ai API key in .env file.")

    logger.info(f"Editing image with fal model: {model}")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Input images: {[str(p) for p in image_paths]}")
    logger.info(f"Size: {width}x{height}")

    def _edit():
        # Upload images to fal (with retry logic built into _upload_image_to_fal)
        image_urls = []
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Uploading image {idx + 1}/{len(image_paths)}: {image_path.name}")
            url = _upload_image_to_fal(image_path)
            image_urls.append(url)

        logger.info(f"All {len(image_urls)} images uploaded successfully")

        # Call fal edit API
        result = fal_client.subscribe(
            model,
            arguments={
                "prompt": prompt,
                "image_size": {
                    "height": height,
                    "width": width
                },
                "num_images": num_images,
                "max_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "enhance_prompt_mode": enhance_prompt_mode,
                "image_urls": image_urls
            },
            with_logs=True,
            on_queue_update=_on_queue_update,
        )

        logger.info(f"fal edit complete. Seed: {result.get('seed', 'N/A')}")

        # Extract image URL from result
        if not result.get("images") or len(result["images"]) == 0:
            raise Exception("No images returned from fal")

        image_url = result["images"][0]["url"]
        seed = result.get("seed", 0)

        # Download image
        logger.info(f"Downloading edited image from: {image_url}")
        image = _download_image_from_url(image_url)

        logger.info(f"Successfully edited image to {width}x{height}")
        return image, seed

    try:
        return _retry_with_backoff(_edit, max_retries=2, initial_delay=3.0)
    except Exception as e:
        logger.error(f"Failed to edit image with fal after retries: {e}")
        raise


def is_fal_available() -> bool:
    """
    Check if fal_client is installed and available.

    Returns:
        True if fal_client is available, False otherwise
    """
    return FAL_AVAILABLE
