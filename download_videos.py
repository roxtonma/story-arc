"""
Standalone script to download missing videos from metadata.json.

Usage:
    python download_videos.py <session_dir>

Example:
    python download_videos.py outputs/projects/20251120_145313
"""

import sys
import os
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def download_videos_from_metadata(session_dir: str):
    """
    Download missing videos from a session's metadata.json.

    Args:
        session_dir: Path to session directory (e.g., outputs/projects/20251120_145313)
    """
    session_path = Path(session_dir)

    if not session_path.exists():
        logger.error(f"Session directory not found: {session_path}")
        return False

    metadata_path = session_path / "assets" / "videos" / "metadata.json"

    if not metadata_path.exists():
        logger.error(f"metadata.json not found at: {metadata_path}")
        return False

    logger.info(f"Loading configuration...")

    # Load config for Agent 10
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    agent_10_config = config.get("agents", {}).get("agent_10", {})

    # Import Agent 10
    from agents.agent_10_video_dialogue import VideoDialogueAgent
    from core.gemini_client import GeminiClient

    # Initialize Gemini client (not used for downloads, but required for Agent init)
    # Use environment variable or dummy key
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "dummy"

    try:
        gemini_client = GeminiClient(api_key=gemini_api_key, model_name="gemini-3-pro-preview")
    except Exception as e:
        logger.warning(f"Could not initialize Gemini client (not needed for downloads): {e}")
        # Create a minimal mock client
        class MockGeminiClient:
            pass
        gemini_client = MockGeminiClient()

    # Initialize Agent 10
    logger.info(f"Initializing Agent 10...")
    agent = VideoDialogueAgent(
        gemini_client=gemini_client,
        config=agent_10_config,
        session_dir=session_path
    )

    # Download missing videos
    logger.info(f"Checking for missing videos...")
    result = agent.download_missing_videos(metadata_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos: {result.get('total_videos', 0)}")
    logger.info(f"Downloaded: {result.get('downloaded', 0)}")
    logger.info(f"Skipped (already exists): {result.get('skipped', 0)}")
    logger.info(f"Failed: {result.get('failed', 0)}")
    logger.info("=" * 60)

    if result.get("downloaded", 0) > 0:
        logger.info(f"✓ Successfully downloaded {result['downloaded']} videos")

    return result.get("status") == "success"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_dir = sys.argv[1]

    logger.info("=" * 60)
    logger.info("Video Download Recovery Tool")
    logger.info("=" * 60)
    logger.info(f"Session: {session_dir}")
    logger.info("")

    success = download_videos_from_metadata(session_dir)

    if success:
        logger.info("✓ Download complete!")
        sys.exit(0)
    else:
        logger.error("✗ Download failed")
        sys.exit(1)
