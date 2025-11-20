"""
Agent 10: Video Dialogue Generator
Generates videos for parent and child shots using FAL AI's Veo3.1 API.
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


class VideoDialogueAgent(BaseAgent):
    """Agent for generating video dialogues using FAL AI Veo3.1."""

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

        # FAL configuration
        self.fal_api_key = config.get("fal_api_key") or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if not self.fal_api_key:
            raise ValueError("FAL_KEY not found in config or environment (set FAL_KEY environment variable)")

        self.video_resolution = config.get("video_resolution", "1080p")
        self.generate_audio = config.get("generate_audio", True)
        self.aspect_ratio = config.get("aspect_ratio", "auto")
        self.gemini_model = config.get("model", "gemini-3-pro-preview")

        # Import and configure fal_client
        try:
            import fal_client
            self.fal_client = fal_client
            # Configure API key
            os.environ["FAL_KEY"] = self.fal_api_key
        except ImportError:
            raise ImportError("fal_client library not installed. Run: pip install fal-client")

        logger.info(f"{self.agent_name}: Initialized with FAL API for video generation")

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
                "fal_model": "fal-ai/veo3.1/fast/image-to-video"
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
                "fal_model": "fal-ai/veo3.1/fast/image-to-video"
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
        screenplay = shot_details.get("dialogue", "") or shot_details.get("action", "")
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
        production_brief = self._generate_production_brief(
            pil_image=pil_image,
            screenplay=screenplay,
            shot_number=shot_number,
            shot_description=shot_description,
            character_details=character_details,
            shot_type=shot_type
        )

        # Save production brief as JSON
        brief_path = self.briefs_dir / f"{shot_id}_brief.json"
        with open(brief_path, 'w', encoding='utf-8') as f:
            json.dump(production_brief, f, indent=2, ensure_ascii=False)

        # Extract video generation prompt and duration
        video_prompt = production_brief.get("video_production_brief", {}).get(
            "video_generation_prompt",
            "A cinematic shot with subtle movement"
        )
        duration_seconds = production_brief.get("video_production_brief", {}).get(
            "duration_seconds", 6
        )

        # Step 2: Upload image to FAL
        logger.info(f"{self.agent_name}: Uploading image to FAL...")
        image_url = self.fal_client.upload_file(str(image_path))
        logger.debug(f"{self.agent_name}: Image uploaded: {image_url}")

        # Step 3: Generate video using FAL API
        logger.info(f"{self.agent_name}: Calling FAL API for video generation (this may take several minutes)...")
        logger.info(f"{self.agent_name}: Video duration: {duration_seconds}s")
        video_result = self._call_fal_api(
            prompt=video_prompt,
            image_url=image_url,
            duration=f"{duration_seconds}s"
        )

        # Extract video URL
        video_url = video_result.get("video", {}).get("url")
        if not video_url:
            raise ValueError("No video URL in FAL API response")

        # Step 4: Download video to local directory
        video_path = self._download_video(video_url, shot_id)

        # Prepare result
        try:
            brief_rel_path = str(brief_path.relative_to(self.session_dir))
        except ValueError:
            brief_rel_path = str(brief_path)

        result = {
            "shot_id": shot_id,
            "shot_type": shot_type,
            "image_path": shot["image_path"],
            "video_url": video_url,
            "video_path": video_path,
            "production_brief_path": brief_rel_path,
            "duration_seconds": duration_seconds,
            "video_prompt": video_prompt,
            "fal_request_id": video_result.get("request_id", "unknown"),
            "generated_at": datetime.now().isoformat(),
            "status": "success"
        }

        logger.info(f"{self.agent_name}: ✓ Video generated successfully for {shot_id}")
        return result

    def _generate_production_brief(
        self,
        pil_image: Image.Image,
        screenplay: str,
        shot_number: str,
        shot_description: str,
        character_details: str,
        shot_type: str
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

        Returns:
            Production brief dictionary
        """
        # Format the prompt with context
        formatted_prompt = self.prompt_template.format(
            screenplay=screenplay,
            shot_number=shot_number,
            shot_description=shot_description,
            character_details=character_details,
            shot_type=shot_type
        )

        # Call Gemini with image and prompt (expecting YAML output per prompt)
        response = self.client.client.models.generate_content(
            model=self.gemini_model,
            contents=[pil_image, formatted_prompt],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens
            )
        )

        # Parse YAML response with Pydantic validation
        response_text = response.text.strip()

        # Remove markdown code fences if present
        if response_text.startswith("```yaml") or response_text.startswith("```yml"):
            # Remove language-specific fence
            response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[6:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # Parse and validate YAML with Pydantic
        try:
            result = yaml.safe_load(response_text)
            validated = ProductionBriefResponse(**result)
            return validated.model_dump()
        except yaml.YAMLError as e:
            logger.error(f"{self.agent_name}: Invalid YAML from Gemini: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise ValueError(f"Invalid YAML response from Gemini: {str(e)}")
        except ValidationError as e:
            logger.error(f"{self.agent_name}: YAML validation failed: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise ValueError(f"Invalid production brief structure: {str(e)}")

    def _call_fal_api(
        self,
        prompt: str,
        image_url: str,
        duration: str
    ) -> Dict[str, Any]:
        """
        Call FAL AI API to generate video.

        Args:
            prompt: Video generation prompt
            image_url: URL of the starting image
            duration: Duration string (e.g., "6s")

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

        # Call FAL API
        logger.info(f"{self.agent_name}: Submitting to FAL queue...")
        try:
            result = self.fal_client.subscribe(
                "fal-ai/veo3.1/fast/image-to-video",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "aspect_ratio": "auto",
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
