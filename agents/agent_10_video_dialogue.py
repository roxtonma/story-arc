"""
Agent 10: Video Dialogue Generator
Generates videos for parent and child shots using FAL AI Veo3.1 or Vertex AI Veo 3.1 Fast.
Supports automatic aspect ratio detection and image optimization for base64 encoding.
"""

import json
import yaml
import os
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from google.genai import types
from core.validators import ProductionBriefResponse
from utils.image_utils import get_aspect_ratio_from_image, optimize_image_for_base64
from utils.vertex_veo_helper import VertexVeoClient


class VideoDialogueAgent(BaseAgent):
    """Agent for generating video dialogues using FAL AI Veo3.1 or Vertex AI Veo 3.1 Fast."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """
        Initialize Video Dialogue Agent.

        Args:
            gemini_client: Initialized Gemini client
            config: Agent configuration
            session_dir: Session directory path
        """
        super().__init__(gemini_client, config, "agent_10")

        # Session and output directories
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.videos_dir = self.assets_dir / "videos"
        self.briefs_dir = self.assets_dir / "video_briefs"

        # Create output directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.briefs_dir.mkdir(parents=True, exist_ok=True)

        # Video provider configuration
        self.video_provider = config.get("video_provider", "fal").lower()
        self.aspect_ratio = config.get("aspect_ratio", "auto")
        self.max_image_size_mb = config.get("max_image_size_mb", 7.0)
        self.gemini_model = config.get("model", "gemini-3-pro-preview")
        self.max_retries = config.get("max_retries", 3)
        
        # FAL configuration
        if self.video_provider == "fal":
            self.fal_api_key = config.get("fal_api_key") or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
            if not self.fal_api_key:
                raise ValueError("FAL_KEY not found in config or environment (set FAL_KEY environment variable)")
            
            self.video_resolution = config.get("video_resolution", "1080p")
            self.generate_audio = config.get("generate_audio", True)
            
            # Import and configure fal_client
            try:
                import fal_client
                self.fal_client = fal_client
                # Configure API key
                os.environ["FAL_KEY"] = self.fal_api_key
            except ImportError:
                raise ImportError("fal_client library not installed. Run: pip install fal-client")
            
            logger.info(f"{self.agent_name}: Initialized with FAL API for video generation")
        
        # Vertex AI configuration
        elif self.video_provider == "vertex_ai":
            self.vertex_project_id = config.get("vertex_ai_project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.vertex_project_id:
                raise ValueError("vertex_ai_project_id not found in config or GOOGLE_CLOUD_PROJECT environment variable")
            
            self.vertex_location = (
                config.get("vertex_ai_location") or 
                os.getenv("GOOGLE_CLOUD_LOCATION") or 
                os.getenv("GOOGLE_CLOUD_REGION") or 
                "us-central1"
            )
            self.vertex_model_id = config.get("vertex_ai_model_id", "veo-3.1-fast-generate-preview")
            self.vertex_credentials_file = config.get("vertex_ai_credentials_file")
            
            # Initialize Vertex AI client
            try:
                self.vertex_client = VertexVeoClient(
                    project_id=self.vertex_project_id,
                    location=self.vertex_location,
                    model_id=self.vertex_model_id,
                    credentials_file=self.vertex_credentials_file,
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Vertex AI client: {str(e)}")
            
            logger.info(f"{self.agent_name}: Initialized with Vertex AI Veo for video generation")
        
        else:
            raise ValueError(f"Unknown video_provider: {self.video_provider}. Options: 'fal', 'vertex_ai'")

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = [
            "parent_shots", "child_shots", "scene_breakdown",
            "shot_breakdown", "shot_grouping", "character_data"
        ]

        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        # Verify we have shots to process
        if not input_data["parent_shots"] and not input_data["child_shots"]:
            raise ValueError("No shots to process (both parent_shots and child_shots are empty)")

        logger.debug(f"{self.agent_name}: Input validation passed")
        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        required_keys = ["videos", "total_videos", "metadata"]
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Output must contain '{key}'")

        if not isinstance(output_data["videos"], list):
            raise ValueError("'videos' must be a list")

        logger.debug(f"{self.agent_name}: Output validation passed")
        return True

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process all shots and generate videos."""
        logger.info(f"{self.agent_name}: Starting video generation for all shots...")

        # Extract input data
        parent_shots = input_data["parent_shots"]
        child_shots = input_data["child_shots"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        character_data = input_data.get("character_data", {})

        # Create lookup tables
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Get characters list from character_data (agent_5 output)
        characters_list = character_data.get("characters", []) if isinstance(character_data, dict) else []
        characters_by_name = {
            char.get("name"): char
            for char in characters_list
            if isinstance(char, dict)
        }

        # Collect all shots to process
        all_videos = []

        # Initialize metadata.json with progress tracking
        self._initialize_metadata(parent_shots, child_shots)

        # Process parent shots first
        logger.info(f"{self.agent_name}: Processing {len(parent_shots)} parent shots...")
        for parent_shot in parent_shots:
            try:
                video_data = self._generate_video_for_shot(
                    shot=parent_shot,
                    shot_type="parent",
                    shots_by_id=shots_by_id,
                    characters_by_name=characters_by_name
                )
                all_videos.append(video_data)
                self._update_metadata(video_data)  # Incremental save
                logger.info(f"✓ Generated video for parent shot: {parent_shot['shot_id']}")
            except Exception as e:
                logger.error(f"✗ Failed to generate video for parent shot {parent_shot['shot_id']}: {str(e)}")
                # Continue with other shots instead of failing completely
                failed_video_data = {
                    "shot_id": parent_shot["shot_id"],
                    "shot_type": "parent",
                    "status": "failed",
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                }
                all_videos.append(failed_video_data)
                self._update_metadata(failed_video_data)  # Incremental save

        # Process child shots
        logger.info(f"{self.agent_name}: Processing {len(child_shots)} child shots...")
        for child_shot in child_shots:
            try:
                video_data = self._generate_video_for_shot(
                    shot=child_shot,
                    shot_type="child",
                    shots_by_id=shots_by_id,
                    characters_by_name=characters_by_name
                )
                all_videos.append(video_data)
                self._update_metadata(video_data)  # Incremental save
                logger.info(f"✓ Generated video for child shot: {child_shot['shot_id']}")
            except Exception as e:
                logger.error(f"✗ Failed to generate video for child shot {child_shot['shot_id']}: {str(e)}")
                # Continue with other shots
                failed_video_data = {
                    "shot_id": child_shot["shot_id"],
                    "shot_type": "child",
                    "status": "failed",
                    "error": str(e),
                    "generated_at": datetime.now().isoformat()
                }
                all_videos.append(failed_video_data)
                self._update_metadata(failed_video_data)  # Incremental save

        # Create output structure
        successful_videos = [v for v in all_videos if v.get("status") != "failed"]
        failed_videos = [v for v in all_videos if v.get("status") == "failed"]

        output = {
            "videos": all_videos,
            "total_videos": len(all_videos),
            "successful_videos": len(successful_videos),
            "failed_videos": len(failed_videos),
            "metadata": {
                "session_id": self.session_dir.name,
                "generated_at": datetime.now().isoformat(),
                "total_parent_videos": len(parent_shots),
                "total_child_videos": len(child_shots),
                "model_used": self.gemini_model,
                "video_provider": self.video_provider,
                "video_model": "fal-ai/veo3.1/fast/image-to-video" if self.video_provider == "fal" else "veo-3.1-fast"
            }
        }

        # Save metadata (final write for backwards compatibility)
        metadata_path = self.videos_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        # Finalize metadata with completion status
        self._finalize_metadata(metadata_path)

        logger.info(
            f"{self.agent_name}: Completed video generation. "
            f"Success: {len(successful_videos)}, Failed: {len(failed_videos)}"
        )

        return output

    def _initialize_metadata(self, parent_shots: List[Dict], child_shots: List[Dict]) -> None:
        """
        Initialize metadata.json with empty structure at start of video generation.

        Args:
            parent_shots: List of parent shot data
            child_shots: List of child shot data
        """
        metadata_path = self.videos_dir / "metadata.json"

        initial_metadata = {
            "videos": [],
            "total_videos": len(parent_shots) + len(child_shots),
            "successful_videos": 0,
            "failed_videos": 0,
            "status": "in_progress",
            "metadata": {
                "session_id": self.session_dir.name,
                "started_at": datetime.now().isoformat(),
                "total_parent_videos": len(parent_shots),
                "total_child_videos": len(child_shots),
                "model_used": self.gemini_model,
                "video_provider": self.video_provider,
                "video_model": "fal-ai/veo3.1/fast/image-to-video" if self.video_provider == "fal" else "veo-3.1-fast"
            }
        }

        # Atomic write: write to temp file, then rename
        temp_path = metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(initial_metadata, f, indent=2)
        temp_path.replace(metadata_path)

        logger.info(f"{self.agent_name}: Initialized metadata.json with 0/{initial_metadata['total_videos']} videos")

    def _update_metadata(self, video_data: Dict[str, Any]) -> None:
        """
        Append newly completed video to metadata.json (incremental update).

        Args:
            video_data: Video result data (success or failure)
        """
        metadata_path = self.videos_dir / "metadata.json"

        # Read current metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            # Fallback if metadata doesn't exist
            metadata = {
                "videos": [],
                "total_videos": 0,
                "successful_videos": 0,
                "failed_videos": 0,
                "status": "in_progress",
                "metadata": {}
            }

        # Append new video
        metadata["videos"].append(video_data)

        # Update counts
        if video_data.get("status") == "success":
            metadata["successful_videos"] = metadata.get("successful_videos", 0) + 1
        else:
            metadata["failed_videos"] = metadata.get("failed_videos", 0) + 1

        # Atomic write
        temp_path = metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(metadata_path)

        completed = metadata["successful_videos"] + metadata["failed_videos"]
        logger.debug(
            f"{self.agent_name}: Updated metadata.json - "
            f"{completed}/{metadata['total_videos']} videos complete"
        )

    def _finalize_metadata(self, metadata_path: Path) -> None:
        """
        Update metadata.json with final completion status.

        Args:
            metadata_path: Path to metadata.json
        """
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Update status and completion time
            metadata["status"] = "completed"
            metadata["metadata"]["completed_at"] = datetime.now().isoformat()

            # Atomic write
            temp_path = metadata_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(metadata_path)

            logger.info(f"{self.agent_name}: Finalized metadata.json - status: completed")

    def download_missing_videos(self, metadata_path: Path = None) -> Dict[str, Any]:
        """
        Download videos from metadata.json if local files are missing.
        Useful for recovery after crashes or if video files were deleted.

        Args:
            metadata_path: Path to metadata.json (optional, defaults to self.videos_dir/metadata.json)

        Returns:
            Summary dict with download statistics
        """
        if metadata_path is None:
            metadata_path = self.videos_dir / "metadata.json"

        if not metadata_path.exists():
            logger.error(f"{self.agent_name}: metadata.json not found at {metadata_path}")
            return {
                "status": "error",
                "message": "metadata.json not found",
                "downloaded": 0,
                "skipped": 0,
                "failed": 0
            }

        logger.info(f"{self.agent_name}: Loading metadata from {metadata_path}")

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        videos = metadata.get("videos", [])
        if not videos:
            logger.info(f"{self.agent_name}: No videos found in metadata")
            return {
                "status": "success",
                "message": "No videos in metadata",
                "downloaded": 0,
                "skipped": 0,
                "failed": 0
            }

        logger.info(f"{self.agent_name}: Found {len(videos)} videos in metadata")

        # Check which videos need downloading
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0
        updated = False

        for i, video in enumerate(videos):
            shot_id = video.get("shot_id", f"SHOT_{i}")
            video_url = video.get("video_url")
            video_path = video.get("video_path")
            status = video.get("status")

            # Skip failed videos or videos without URLs
            if status == "failed" or not video_url:
                skipped_count += 1
                continue

            # Check if video file exists locally
            if video_path:
                full_path = self.session_dir / video_path
                if full_path.exists():
                    logger.debug(f"{self.agent_name}: {shot_id} - Video file already exists, skipping")
                    skipped_count += 1
                    continue

            # Download missing video
            logger.info(f"{self.agent_name}: Downloading missing video for {shot_id}...")
            try:
                new_video_path = self._download_video(video_url, shot_id)
                video["video_path"] = new_video_path
                downloaded_count += 1
                updated = True
                logger.info(f"{self.agent_name}: ✓ Downloaded {shot_id}")
            except Exception as e:
                logger.error(f"{self.agent_name}: ✗ Failed to download {shot_id}: {str(e)}")
                failed_count += 1

        # Save updated metadata if any videos were downloaded
        if updated:
            logger.info(f"{self.agent_name}: Updating metadata.json with new video paths...")
            temp_path = metadata_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            temp_path.replace(metadata_path)
            logger.info(f"{self.agent_name}: Metadata updated successfully")

        # Summary
        summary = {
            "status": "success",
            "total_videos": len(videos),
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "failed": failed_count
        }

        logger.info(
            f"{self.agent_name}: Download complete - "
            f"Downloaded: {downloaded_count}, Skipped: {skipped_count}, Failed: {failed_count}"
        )

        return summary

    def generate_single_shot_video(
        self,
        shot_id: str,
        shot_type: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate video for a single shot (for UI shot-level generation).

        Args:
            shot_id: Shot identifier (e.g., "SHOT_1_1")
            shot_type: "parent" or "child"
            input_data: Complete input data from session (containing all agent outputs)

        Returns:
            Video generation result for the single shot

        Raises:
            ValueError: If shot not found or invalid input
        """
        logger.info(f"{self.agent_name}: Generating single shot video for {shot_id} ({shot_type})")

        # Extract required data from input
        parent_shots = input_data.get("parent_shots", [])
        child_shots = input_data.get("child_shots", [])
        shot_breakdown = input_data.get("shot_breakdown", {})
        character_data = input_data.get("character_data", {})

        # Create lookup tables
        shots_by_id = {}
        if shot_breakdown.get("shots"):
            for shot in shot_breakdown["shots"]:
                shots_by_id[shot["shot_id"]] = shot

        characters_by_name = {}
        if character_data.get("characters"):
            for char in character_data["characters"]:
                characters_by_name[char["name"]] = char

        # Find the target shot
        target_shot = None
        if shot_type == "parent":
            target_shot = next((s for s in parent_shots if s["shot_id"] == shot_id), None)
        elif shot_type == "child":
            target_shot = next((s for s in child_shots if s["shot_id"] == shot_id), None)
        else:
            raise ValueError(f"Invalid shot_type: {shot_type}. Must be 'parent' or 'child'")

        if not target_shot:
            raise ValueError(f"Shot {shot_id} not found in {shot_type} shots")

        # Generate video for this shot
        try:
            video_data = self._generate_video_for_shot(
                shot=target_shot,
                shot_type=shot_type,
                shots_by_id=shots_by_id,
                characters_by_name=characters_by_name
            )

            # Update metadata with single shot
            self._update_single_shot_metadata(video_data)

            logger.info(f"{self.agent_name}: Successfully generated video for {shot_id}")
            return video_data

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to generate video for {shot_id}: {str(e)}")
            # Return failed video data
            failed_video_data = {
                "shot_id": shot_id,
                "shot_type": shot_type,
                "status": "failed",
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
            self._update_single_shot_metadata(failed_video_data)
            return failed_video_data

    def _update_single_shot_metadata(self, video_data: Dict[str, Any]) -> None:
        """
        Update metadata.json with a single shot video (used for UI-triggered generation).

        Args:
            video_data: Video result data for single shot
        """
        metadata_path = self.videos_dir / "metadata.json"

        # Read current metadata or create new
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            # Initialize metadata if doesn't exist
            metadata = {
                "videos": [],
                "total_videos": 0,
                "successful_videos": 0,
                "failed_videos": 0,
                "status": "in_progress",
                "metadata": {
                    "session_id": self.session_dir.name,
                    "video_provider": self.video_provider
                }
            }

        # Check if this shot already exists in metadata
        existing_index = None
        for i, video in enumerate(metadata["videos"]):
            if video.get("shot_id") == video_data.get("shot_id"):
                existing_index = i
                break

        # Update or append
        if existing_index is not None:
            # Replace existing entry
            metadata["videos"][existing_index] = video_data
            logger.info(f"Updated existing video entry for {video_data.get('shot_id')}")
        else:
            # Add new entry
            metadata["videos"].append(video_data)
            metadata["total_videos"] = len(metadata["videos"])
            logger.info(f"Added new video entry for {video_data.get('shot_id')}")

        # Recalculate success/failure counts
        successful = sum(1 for v in metadata["videos"] if v.get("status") != "failed")
        failed = sum(1 for v in metadata["videos"] if v.get("status") == "failed")
        metadata["successful_videos"] = successful
        metadata["failed_videos"] = failed

        # Atomic write
        temp_path = metadata_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        temp_path.replace(metadata_path)

    def _download_video(self, video_url: str, shot_id: str) -> str:
        """
        Download video file from FAL URL to local videos/ directory.

        Args:
            video_url: FAL CDN URL for the video
            shot_id: Shot identifier (e.g., "SHOT_1_1")

        Returns:
            Relative path to downloaded video file (e.g., "assets/videos/SHOT_1_1.mp4")
        """
        # Define local video path
        video_filename = f"{shot_id}.mp4"
        video_path = self.videos_dir / video_filename

        logger.info(f"{self.agent_name}: Downloading video from FAL CDN...")
        logger.debug(f"{self.agent_name}: URL: {video_url}")

        try:
            # Download video with streaming to handle large files
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()

            # Write to file in chunks
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Get relative path from session directory
            try:
                rel_path = video_path.relative_to(self.session_dir)
            except ValueError:
                rel_path = video_path

            logger.info(f"{self.agent_name}: Video downloaded successfully to {video_filename}")
            return str(rel_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"{self.agent_name}: Failed to download video: {str(e)}")
            raise ValueError(f"Video download failed: {str(e)}")

    def _generate_video_for_shot(
        self,
        shot: Dict[str, Any],
        shot_type: str,
        shots_by_id: Dict[str, Any],
        characters_by_name: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate video for a single shot.

        Args:
            shot: Shot data (parent or child)
            shot_type: "parent" or "child"
            shots_by_id: Lookup table for shot details
            characters_by_name: Lookup table for character details

        Returns:
            Video generation result data
        """
        shot_id = shot["shot_id"]
        logger.info(f"{self.agent_name}: Processing {shot_type} shot: {shot_id}")

        # Get shot details from shot_breakdown
        shot_details = shots_by_id.get(shot_id, {})
        
        # Extract shot information
        shot_dialogue = (shot_details.get("dialogue") or "").strip()
        screenplay = shot_dialogue or shot_details.get("action", "")
        shot_number = shot_details.get("shot_number", shot_id)
        shot_description = shot_details.get("description", "") or shot_details.get("shot_type", "")
        characters_in_shot = shot_details.get("characters", [])

        # Build character details string
        character_details_list = []
        for char_name in characters_in_shot:
            char_info = characters_by_name.get(char_name, {})
            if char_info:
                character_details_list.append(
                    f"{char_name}: {char_info.get('description', 'No description')}"
                )
        character_details = " | ".join(character_details_list) if character_details_list else "No characters"

        # Load the shot image
        image_path = self.session_dir / shot["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(image_path)

        # Step 1: Generate video production brief using Gemini
        logger.info(f"{self.agent_name}: Generating production brief with Gemini...")
        
        # Get shot details for duration
        shot_details_full = shot_details.get("shot_description", "") or shot_details.get("first_frame", "")
        
        production_brief = self._generate_production_brief(
            pil_image=pil_image,
            screenplay=screenplay,
            shot_number=shot_number,
            shot_description=shot_description,
            character_details=character_details,
            shot_type=shot_type,
            start_frame=shot_details_full,  # Add missing parameter
            shot_dialogue=shot_dialogue,
        )

        # Save production brief as JSON
        brief_path = self.briefs_dir / f"{shot_id}_brief.json"
        with open(brief_path, 'w', encoding='utf-8') as f:
            json.dump(production_brief, f, indent=2, ensure_ascii=False)

        # Extract requested duration from production brief
        raw_duration = production_brief.get("video_production_brief", {}).get(
            "duration_seconds", 6
        )

        # Normalize to an integer number of seconds
        try:
            requested_duration_seconds = int(round(float(raw_duration)))
        except (TypeError, ValueError):
            logger.warning(
                f"{self.agent_name}: Invalid duration_seconds in brief "
                f"({raw_duration!r}), defaulting to 6s"
            )
            requested_duration_seconds = 6

        # For FAL, only 4s, 6s, 8s are supported. We normalize globally so both
        # backends are consistent and clips stay within the 4–8s window.
        if requested_duration_seconds <= 5:
            effective_duration_seconds = 4
        elif requested_duration_seconds <= 7:
            effective_duration_seconds = 6
        else:
            effective_duration_seconds = 8

        logger.info(
            f"{self.agent_name}: Duration from brief: {requested_duration_seconds}s, "
            f"normalized to {effective_duration_seconds}s for video generation"
        )

        # Detect aspect ratio from image if set to "auto"
        if self.aspect_ratio == "auto":
            detected_aspect_ratio = get_aspect_ratio_from_image(image_path)
            logger.info(f"{self.agent_name}: Detected aspect ratio: {detected_aspect_ratio}")
        else:
            detected_aspect_ratio = self.aspect_ratio
            logger.debug(f"{self.agent_name}: Using configured aspect ratio: {detected_aspect_ratio}")

        # Step 2: Generate video based on provider (with retries)
        video_result_metadata = {}
        
        for attempt in range(self.max_retries):
            try:
                if self.video_provider == "fal":
                    # FAL workflow - upload image and call API
                    logger.info(f"{self.agent_name}: Uploading image to FAL...")
                    image_url = self.fal_client.upload_file(str(image_path))
                    logger.debug(f"{self.agent_name}: Image uploaded: {image_url}")

                    logger.info(f"{self.agent_name}: Calling FAL API for video generation (this may take several minutes)...")
                    logger.info(
                        f"{self.agent_name}: Video duration (FAL): "
                        f"{effective_duration_seconds}s, aspect ratio: {detected_aspect_ratio}"
                    )
                    
                    video_result = self._call_fal_api(
                        prompt=production_brief,  # Send full production brief
                        image_url=image_url,
                        duration=f"{effective_duration_seconds}s",
                        aspect_ratio=detected_aspect_ratio
                    )

                    # Extract video URL
                    video_url = video_result.get("video", {}).get("url")
                    if not video_url:
                        raise ValueError("No video URL in FAL API response")

                    # Download video to local directory
                    video_path = self._download_video(video_url, shot_id)
                    
                    video_result_metadata = {
                        "video_url": video_url,
                        "video_path": video_path,
                        "fal_request_id": video_result.get("request_id", "unknown")
                    }
                
                elif self.video_provider == "vertex_ai":
                    # Vertex AI workflow - optimize and encode image
                    logger.info(f"{self.agent_name}: Optimizing image for base64 encoding...")
                    image_base64, final_size = optimize_image_for_base64(
                        image_path=image_path,
                        max_size_mb=self.max_image_size_mb,
                        output_format="JPEG",
                        quality=85
                    )
                    logger.info(f"{self.agent_name}: Image optimized: {final_size / (1024*1024):.2f} MB")

                    logger.info(f"{self.agent_name}: Calling Vertex AI Veo for video generation (this may take several minutes)...")
                    logger.info(
                        f"{self.agent_name}: Video duration (Vertex AI): "
                        f"{effective_duration_seconds}s, aspect_ratio: {detected_aspect_ratio}"
                    )
                    
                    video_result = self._call_vertex_ai_veo(
                        prompt=production_brief,  # Send full production brief
                        image_base64=image_base64,
                        duration_seconds=effective_duration_seconds,
                        aspect_ratio=detected_aspect_ratio
                    )

                    # Save video from base64
                    video_filename = f"{shot_id}.mp4"
                    video_path_full = self.videos_dir / video_filename
                    self.vertex_client.save_video_from_base64(
                        video_result["video_base64"],
                        video_path_full
                    )
                    
                    # Get relative path
                    try:
                        video_path = str(video_path_full.relative_to(self.session_dir))
                    except ValueError:
                        video_path = str(video_path_full)
                    
                    video_result_metadata = {
                        "video_url": None,  # No URL for Vertex AI
                        "video_path": video_path,
                        "vertex_operation": video_result.get("operation_name", "N/A")
                    }
                
                else:
                    raise ValueError(f"Unknown video_provider: {self.video_provider}")
                
                # Success - break loop
                break
                
            except Exception as e:
                logger.warning(f"{self.agent_name}: Video generation failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise

        # Prepare result
        try:
            brief_rel_path = str(brief_path.relative_to(self.session_dir))
        except ValueError:
            brief_rel_path = str(brief_path)

        result = {
            "shot_id": shot_id,
            "shot_type": shot_type,
            "image_path": shot["image_path"],
            "production_brief_path": brief_rel_path,
            "requested_duration_seconds": requested_duration_seconds,
            "duration_seconds": effective_duration_seconds,
            "aspect_ratio": detected_aspect_ratio,
            "video_provider": self.video_provider,
            "generated_at": datetime.now().isoformat(),
            "status": "success"
        }
        
        # Add provider-specific metadata
        result.update(video_result_metadata)

        # Write per-shot JSON metadata next to the video file
        try:
            video_path_value = result.get("video_path")
            if video_path_value:
                # Resolve full video path and construct JSON path with same stem
                video_full_path = self.session_dir / video_path_value
                video_json_path = video_full_path.with_suffix(".json")

                shot_meta = {
                    "shot_id": shot_id,
                    "shot_type": shot_type,
                    "image_path": shot["image_path"],
                    "video_path": video_path_value,
                    "video_provider": self.video_provider,
                    "requested_duration_seconds": requested_duration_seconds,
                    "duration_seconds": effective_duration_seconds,
                    "aspect_ratio": detected_aspect_ratio,
                    "production_brief_path": brief_rel_path,
                    # Store the exact structured prompt sent to the video backend
                    "prompt_used": production_brief,
                }

                video_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(video_json_path, "w", encoding="utf-8") as jf:
                    json.dump(shot_meta, jf, indent=2, ensure_ascii=False)

                logger.debug(
                    f"{self.agent_name}: Wrote per-shot metadata JSON to "
                    f"{video_json_path.relative_to(self.session_dir)}"
                )
            else:
                logger.warning(
                    f"{self.agent_name}: No video_path for {shot_id}, "
                    "skipping per-shot JSON metadata file"
                )
        except Exception as e:
            logger.warning(
                f"{self.agent_name}: Failed to write per-shot JSON metadata for "
                f"{shot_id}: {e}"
            )

        logger.info(f"{self.agent_name}: ✓ Video generated successfully for {shot_id}")
        return result

    def _dialogue_is_covered(self, shot_dialogue: str, brief: Dict[str, Any]) -> bool:
        """
        Check whether the Agent 3 dialogue line appears in at least one audio_event.

        This is a soft guardrail to ensure that when Agent 3 provides a dialogue
        line like `LOKI: Must you announce your thirst`, the production brief
        actually speaks it at least once in the temporal_action_plan.
        """
        dialog = (shot_dialogue or "").strip()
        if not dialog:
            # Nothing to enforce
            return True

        # Try to separate speaker and line (e.g. 'LOKI: Must you announce your thirst')
        speaker, _, line = dialog.partition(":")
        speaker = speaker.strip()
        line = line.strip()

        try:
            vp = brief.get("video_production_brief", {})
            segments = vp.get("temporal_action_plan", []) or []
        except AttributeError:
            return False

        dialog_lower = dialog.lower()
        line_lower = line.lower() if line else ""
        speaker_lower = speaker.lower() if speaker else ""

        for seg in segments:
            audio = (seg.get("audio_event") or "").strip()
            audio_lower = audio.lower()

            # Strongest case: full dialogue line appears
            if dialog_lower and dialog_lower in audio_lower:
                return True

            # Fallback: core line (after colon) plus speaker name both appear
            if line_lower and line_lower in audio_lower:
                if not speaker_lower or speaker_lower in audio_lower:
                    return True

        return False

    def _generate_production_brief(
        self,
        pil_image: Image.Image,
        screenplay: str,
        shot_number: str,
        shot_description: str,
        character_details: str,
        shot_type: str,
        start_frame: str,
        shot_dialogue: str,
    ) -> Dict[str, Any]:
        """
        Generate video production brief using Gemini.

        Args:
            pil_image: PIL Image of the shot
            screenplay: Dialogue or action text
            shot_number: Shot number/ID
            shot_description: Shot description
            character_details: Character details string
            shot_type: "parent" or "child"
            start_frame: Description of the starting frame

        Returns:
            Production brief dictionary
        """
        # Check if prompt template was loaded
        if not self.prompt_template:
            raise ValueError(f"Prompt template not loaded. Check that {self.prompt_file} exists.")

        # Format the prompt with context (clip_duration will be filled by the model)
        try:
            formatted_prompt = self.prompt_template.format(
                screenplay=screenplay,
                shot_number=shot_number,
                shot_description=shot_description,
                character_details=character_details,
                shot_type=shot_type,
                clip_duration="4-8 seconds",  # Provide placeholder value
                start_frame=start_frame
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in prompt template: {e}")

        # Retry logic
        max_retries = self.max_retries
        last_error = None
        last_valid_brief = None
        
        for attempt in range(max_retries):
            try:
                # For retries, append explicit feedback if dialogue was omitted previously
                if attempt == 0 or not shot_dialogue.strip():
                    prompt_for_attempt = formatted_prompt
                else:
                    prompt_for_attempt = (
                        formatted_prompt
                        + "\n\nIMPORTANT: The input dialogue line below must appear "
                          "exactly once in one of the audio_event fields in "
                          "temporal_action_plan. Your previous brief omitted this line "
                          "or made the shot fully silent. Regenerate the JSON, keeping "
                          "the same visual plan, but include that spoken line once in "
                          "the appropriate time segment.\n\n"
                        + shot_dialogue
                    )

                # Call Gemini with image and prompt (expecting JSON output per prompt)
                response = self.client.client.models.generate_content(
                    model=self.gemini_model,
                    contents=[pil_image, prompt_for_attempt],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json"
                    )
                )

                # Parse JSON response with Pydantic validation
                response_text = response.text

                if not response_text:
                    if attempt < max_retries - 1:
                        logger.warning(f"{self.agent_name}: Empty response from Gemini (attempt {attempt+1}/{max_retries})")
                        continue
                    raise ValueError("Empty response from Gemini")

                response_text = response_text.strip()

                # Remove markdown code fences if present
                if response_text.startswith("```json"):
                    # Remove language-specific fence
                    response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                response_text = response_text.strip()

                # Parse and validate JSON with Pydantic
                result = json.loads(response_text)
                validated = ProductionBriefResponse(**result)
                brief_dict = validated.model_dump()

                # If there is no dialogue for this shot, accept immediately
                if not shot_dialogue.strip():
                    return brief_dict

                # Enforce that the dialogue line from Agent 3 appears in at least
                # one audio_event. Soft-fail after all retries.
                if self._dialogue_is_covered(shot_dialogue, brief_dict):
                    return brief_dict

                # Dialogue missing from brief
                last_valid_brief = brief_dict
                logger.warning(
                    f"{self.agent_name}: Brief missing dialogue line for shot "
                    f"{shot_number!r} (attempt {attempt+1}/{max_retries}). "
                    f"Expected line: {shot_dialogue!r}"
                )

                # If this was the last attempt, soft-fail and return the best effort
                if attempt == max_retries - 1:
                    logger.warning(
                        f"{self.agent_name}: Soft failure - using brief without explicit "
                        f"dialogue after {max_retries} attempts"
                    )
                    return last_valid_brief or brief_dict

                # Otherwise, retry with explicit feedback appended to the prompt
                continue

            except (json.JSONDecodeError, ValidationError) as e:
                # Hard failure: invalid JSON/structure, keep existing behaviour
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"{self.agent_name}: Failed to parse brief "
                        f"(attempt {attempt+1}/{max_retries}): {str(e)}"
                    )
                    continue
            except Exception as e:
                # Hard failure: API or other runtime error
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"{self.agent_name}: Generation error "
                        f"(attempt {attempt+1}/{max_retries}): {str(e)}"
                    )
                    continue

        # If we get here, all retries failed
        logger.error(f"{self.agent_name}: Failed to generate valid production brief after {max_retries} attempts")
        if last_error:
            raise ValueError(f"Failed to generate production brief: {str(last_error)}")
        raise ValueError("Failed to generate production brief (unknown error)")

    def _call_fal_api(
        self,
        prompt: Dict[str, Any],
        image_url: str,
        duration: str,
        aspect_ratio: str
    ) -> Dict[str, Any]:
        """
        Call FAL AI API to generate video.

        Args:
            prompt: Full production brief dictionary (from Gemini)
            image_url: URL of the starting image
            duration: Duration string (e.g., "6s")
            aspect_ratio: Aspect ratio string (e.g., "16:9")

        Returns:
            FAL API response dictionary
        """
        def on_queue_update(update):
            """Handle queue updates from FAL."""
            if isinstance(update, self.fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"{self.agent_name}: FAL - {log.get('message', '')}")
            elif isinstance(update, self.fal_client.Queued):
                logger.info(f"{self.agent_name}: FAL - Request queued, position: {getattr(update, 'position', 'unknown')}")

        # Convert production brief to JSON string for FAL
        prompt_text = json.dumps(prompt, indent=2)
        logger.debug(f"{self.agent_name}: Sending full production brief to FAL ({len(prompt_text)} chars)")

        # Call FAL API
        logger.info(f"{self.agent_name}: Submitting to FAL queue...")
        try:
            result = self.fal_client.subscribe(
                "fal-ai/veo3.1/fast/image-to-video",
                arguments={
                    "prompt": prompt_text,
                    "image_url": image_url,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "generate_audio": self.generate_audio,
                    "resolution": self.video_resolution
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            logger.info(f"{self.agent_name}: FAL - Video generation completed!")
            return result
        except Exception as e:
            logger.error(f"{self.agent_name}: FAL API error: {str(e)}")
            raise

    def _call_vertex_ai_veo(
        self,
        prompt: Dict[str, Any],
        image_base64: str,
        duration_seconds: int,
        aspect_ratio: str
    ) -> Dict[str, Any]:
        """
        Call Vertex AI Veo API to generate video.

        Args:
            prompt: Full production brief dictionary (from Gemini)
            image_base64: Base64-encoded starting image
            duration_seconds: Effective duration in seconds (already normalized)
            aspect_ratio: Aspect ratio string (e.g., "16:9")

        Returns:
            Vertex AI API response dictionary with video_base64
        """
        logger.info(f"{self.agent_name}: Submitting to Vertex AI Veo...")
        try:
            result = self.vertex_client.generate_video(
                prompt=prompt,
                image_base64=image_base64,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
            )
            logger.info(f"{self.agent_name}: Vertex AI Veo - Video generation completed!")
            return result
        except Exception as e:
            logger.error(f"{self.agent_name}: Vertex AI Veo error: {str(e)}")
            raise
