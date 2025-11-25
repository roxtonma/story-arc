"""
Agent 11: Intelligent Video Editor

Transforms individual AI-generated video clips into a cinematic final product
using audio intelligence (WhisperX), narrative awareness (Gemini), and
sophisticated editing techniques (J/L cuts, dynamic pacing).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from agents.base_agent import BaseAgent
from agents.utils.audio_analyzer import AudioAnalyzer
from agents.utils.ffmpeg_builder import FFmpegBuilder


class VideoEditAgent(BaseAgent):
    """
    Agent 11: Intelligent Video Editor

    Pipeline:
    1. Analyze audio from all video clips using WhisperX
    2. Generate Edit Decision List (EDL) using Gemini 2.5 Flash
    3. Execute edits per scene using FFmpeg
    4. Assemble master timeline
    5. Export final video(s)
    """

    def __init__(
        self,
        gemini_client,
        config: Dict[str, Any],
        session_dir: Path
    ):
        """
        Initialize VideoEditAgent.

        Args:
            gemini_client: Gemini client instance
            config: Agent 11 configuration dictionary
            session_dir: Path to session directory
        """
        super().__init__(gemini_client, config, "agent_11")

        # Session directory
        self.session_dir = Path(session_dir)
        self.videos_dir = self.session_dir / "assets" / "videos"
        self.edit_output_dir = self.session_dir / "assets" / "edited"
        self.edit_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize utilities
        try:
            self.audio_analyzer = AudioAnalyzer(config)
            logger.info(f"{self.agent_name}: AudioAnalyzer initialized")
        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to initialize AudioAnalyzer: {e}")
            raise

        try:
            self.ffmpeg_builder = FFmpegBuilder(config)
            logger.info(f"{self.agent_name}: FFmpegBuilder initialized")
        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to initialize FFmpegBuilder: {e}")
            raise

        # Configuration
        self.max_edl_retries = config.get("max_edl_retries", 3)
        self.use_heuristic_fallback = config.get("use_heuristic_fallback", True)
        self.scene_fade_duration = config.get("scene_fade_duration", 1.0)

        logger.info(f"{self.agent_name}: Initialization complete")

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for Agent 11.

        Args:
            input_data: Input dictionary with videos and metadata

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_keys = ["videos", "scene_breakdown", "shot_breakdown"]

        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required input key: {key}")

        videos = input_data.get("videos", [])
        if not videos:
            raise ValueError("No videos provided for editing")

        # Check if video files exist
        missing_videos = []
        for video_info in videos:
            video_path = self.session_dir / video_info.get("video_path", "")
            if not video_path.exists():
                missing_videos.append(video_info.get("shot_id", "unknown"))

        if missing_videos:
            raise ValueError(f"Missing video files for shots: {', '.join(missing_videos)}")

        logger.info(f"{self.agent_name}: Input validation passed ({len(videos)} videos)")
        return True

    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output data from Agent 11.

        Args:
            output_data: Output dictionary with edited videos

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_keys = ["master_video_path", "scene_videos", "edit_timeline"]

        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")

        # Verify master video exists
        master_path = self.session_dir / output_data["master_video_path"]
        if not master_path.exists():
            raise ValueError(f"Master video not found: {master_path}")

        logger.info(f"{self.agent_name}: Output validation passed")
        return True

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing pipeline for Agent 11.

        Args:
            input_data: Input dictionary with videos and metadata

        Returns:
            Dictionary with edited video outputs
        """
        logger.info(f"{self.agent_name}: Starting video editing pipeline")

        # Extract inputs
        videos = input_data["videos"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]

        logger.info(f"{self.agent_name}: Processing {len(videos)} video clips")

        # Phase 1: Audio Intelligence
        logger.info(f"{self.agent_name}: Phase 1 - Audio Analysis")
        audio_metadata = self.audio_analyzer.analyze_video_batch(videos, self.session_dir)

        # Save audio metadata
        audio_metadata_path = self.edit_output_dir / "audio_metadata.json"
        self.audio_analyzer.save_metadata(audio_metadata, audio_metadata_path)

        # Phase 2: Generate Edit Decision List using Gemini
        logger.info(f"{self.agent_name}: Phase 2 - Generate Edit Decision List")
        edit_timeline = self._generate_edit_timeline(
            videos,
            audio_metadata,
            scene_breakdown,
            shot_breakdown
        )

        # Save EDL
        edl_path = self.edit_output_dir / "edit_decision_list.json"
        with open(edl_path, 'w', encoding='utf-8') as f:
            json.dump(edit_timeline, f, indent=2, ensure_ascii=False)
        logger.info(f"{self.agent_name}: EDL saved to {edl_path}")

        # Phase 3: Edit Scene Videos
        logger.info(f"{self.agent_name}: Phase 3 - Edit Scene Videos")
        scene_videos = self._edit_all_scenes(edit_timeline, videos, audio_metadata)

        # Phase 4: Assemble Master Timeline
        logger.info(f"{self.agent_name}: Phase 4 - Assemble Master Timeline")
        master_video = self._assemble_master_timeline(scene_videos)

        # Phase 5: Compile Output
        total_duration = self.ffmpeg_builder.get_video_duration(master_video)

        output = {
            "master_video_path": str(master_video.relative_to(self.session_dir)),
            "scene_videos": scene_videos,
            "edit_timeline": edit_timeline,
            "total_duration": round(total_duration, 2),
            "edit_metadata": {
                "scenes_edited": len(scene_videos),
                "total_shots": len(videos),
                "audio_analysis_completed": len(audio_metadata),
                "editing_method": edit_timeline.get("editing_method", "gemini_edl")
            }
        }

        logger.info(
            f"{self.agent_name}: Editing complete - "
            f"{output['total_duration']}s ({len(scene_videos)} scenes)"
        )

        return output

    def _generate_edit_timeline(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate Edit Decision List using Gemini 2.5 Flash.

        Args:
            videos: List of video metadata
            audio_metadata: Audio analysis results
            scene_breakdown: Scene structure from Agent 2
            shot_breakdown: Shot details from Agent 3

        Returns:
            Edit timeline dictionary
        """
        # Build context for Gemini
        context = self._build_editing_context(
            videos,
            audio_metadata,
            scene_breakdown,
            shot_breakdown
        )

        # Check if user wants to skip Gemini and use heuristic editing directly
        if self.config.get("skip_gemini_edl", False):
            logger.info(f"{self.agent_name}: Skipping Gemini - using heuristic editing (skip_gemini_edl=true)")
            return self._generate_heuristic_edl(
                videos,
                audio_metadata,
                scene_breakdown,
                shot_breakdown
            )

        # Try Gemini EDL generation with retries
        for attempt in range(self.max_edl_retries):
            try:
                logger.info(
                    f"{self.agent_name}: Gemini EDL generation attempt "
                    f"{attempt + 1}/{self.max_edl_retries}"
                )

                # Format prompt
                # Use direct string replacement to avoid format() issues with JSON braces
                json_str = json.dumps(context, indent=2)
                prompt = self.prompt_template.replace('{input}', json_str)

                # Call Gemini
                response = self.client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )

                # Extract and parse JSON
                edit_timeline = self._extract_json(response)

                # Validate EDL structure
                self._validate_edl(edit_timeline)

                # Add metadata
                edit_timeline["editing_method"] = "gemini_edl"
                edit_timeline["gemini_attempt"] = attempt + 1

                logger.info(f"{self.agent_name}: Gemini EDL generation successful")
                return edit_timeline

            except Exception as e:
                logger.warning(
                    f"{self.agent_name}: Gemini EDL attempt {attempt + 1} failed: {e}"
                )

                if attempt < self.max_edl_retries - 1:
                    logger.info(f"{self.agent_name}: Retrying with simplified prompt...")
                    # On retry, we could simplify the prompt or add error feedback
                    continue

        # All Gemini attempts failed
        logger.warning(
            f"{self.agent_name}: Gemini EDL generation failed after "
            f"{self.max_edl_retries} attempts"
        )

        if self.use_heuristic_fallback:
            logger.info(f"{self.agent_name}: Using heuristic fallback editing")
            return self._generate_heuristic_edl(
                videos,
                audio_metadata,
                scene_breakdown,
                shot_breakdown
            )
        else:
            raise RuntimeError(
                "Gemini EDL generation failed and heuristic fallback is disabled"
            )

    def _build_editing_context(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context dictionary for Gemini EDL generation.

        Args:
            videos: List of video metadata
            audio_metadata: Audio analysis results
            scene_breakdown: Scene structure
            shot_breakdown: Shot details

        Returns:
            Context dictionary for Gemini
        """
        # Group shots by scene
        shots_by_scene = {}
        for shot in shot_breakdown.get("shots", []):
            scene_id = shot.get("scene_id")
            if scene_id not in shots_by_scene:
                shots_by_scene[scene_id] = []

            # Merge shot info with video and audio metadata
            shot_id = shot.get("shot_id")
            video_info = next((v for v in videos if v.get("shot_id") == shot_id), None)
            audio_info = audio_metadata.get(shot_id, {})

            merged_shot = {
                **shot,
                "video_path": video_info.get("video_path") if video_info else None,
                "duration": video_info.get("duration_seconds") if video_info else None,
                "audio_metadata": audio_info.get("dialogue") if audio_info else None
            }

            shots_by_scene[scene_id].append(merged_shot)

        # Build scene list
        scenes = []
        for scene in scene_breakdown.get("scenes", []):
            scene_id = scene.get("scene_id")
            scenes.append({
                "scene_id": scene_id,
                "location": scene.get("location", ""),
                "description": scene.get("description", ""),
                "shots": shots_by_scene.get(scene_id, [])
            })

        return {"scenes": scenes}

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from Gemini response.

        Args:
            response: Gemini response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

    def _validate_edl(self, edit_timeline: Dict[str, Any]):
        """
        Validate Edit Decision List structure.

        Args:
            edit_timeline: EDL dictionary

        Raises:
            ValueError: If EDL structure is invalid
        """
        if "edit_plan" not in edit_timeline:
            raise ValueError("EDL missing 'edit_plan' key")

        edit_plan = edit_timeline["edit_plan"]

        if "scenes" not in edit_plan:
            raise ValueError("Edit plan missing 'scenes' key")

        scenes = edit_plan["scenes"]
        if not isinstance(scenes, list):
            raise ValueError("Scenes must be a list")

        for scene in scenes:
            if "shots" not in scene:
                raise ValueError(f"Scene {scene.get('scene_id')} missing 'shots'")

            for shot in scene["shots"]:
                required_fields = ["shot_id", "edit_type", "trim_start", "trim_end"]
                for field in required_fields:
                    if field not in shot:
                        raise ValueError(
                            f"Shot {shot.get('shot_id', 'unknown')} missing '{field}'"
                        )

        logger.debug(f"{self.agent_name}: EDL validation passed")

    def _edit_all_scenes(
        self,
        edit_timeline: Dict[str, Any],
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Edit all scenes according to EDL.

        Args:
            edit_timeline: Edit Decision List
            videos: Original video metadata
            audio_metadata: Audio analysis results

        Returns:
            List of scene video metadata
        """
        scene_videos = []
        edit_plan = edit_timeline.get("edit_plan", edit_timeline)
        scenes = edit_plan.get("scenes", [])

        for i, scene_edit in enumerate(scenes, 1):
            scene_id = scene_edit.get("scene_id")
            logger.info(f"{self.agent_name}: Editing scene {i}/{len(scenes)}: {scene_id}")

            try:
                scene_video = self._edit_scene(scene_edit, videos)
                scene_videos.append(scene_video)
            except Exception as e:
                logger.error(f"{self.agent_name}: Failed to edit {scene_id}: {e}")
                raise

        return scene_videos

    def _edit_scene(
        self,
        scene_edit: Dict[str, Any],
        videos: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Edit a single scene according to EDL.

        Args:
            scene_edit: Scene edit instructions from EDL
            videos: Original video metadata

        Returns:
            Scene video metadata dictionary
        """
        scene_id = scene_edit.get("scene_id")
        shots = scene_edit.get("shots", [])

        if not shots:
            raise ValueError(f"Scene {scene_id} has no shots")

        # Output path
        output_path = self.edit_output_dir / f"{scene_id}.mp4"

        # Use FFmpegBuilder to execute scene editing
        self.ffmpeg_builder.edit_scene_with_edl(
            shots,
            self.session_dir,
            output_path
        )

        # Get duration
        duration = self.ffmpeg_builder.get_video_duration(output_path)

        return {
            "scene_id": scene_id,
            "video_path": str(output_path.relative_to(self.session_dir)),
            "shot_count": len(shots),
            "duration": round(duration, 2)
        }

    def _assemble_master_timeline(
        self,
        scene_videos: List[Dict[str, Any]]
    ) -> Path:
        """
        Assemble all scene videos into master timeline.

        Args:
            scene_videos: List of scene video metadata

        Returns:
            Path to master video file
        """
        logger.info(f"{self.agent_name}: Assembling master timeline from {len(scene_videos)} scenes")

        # Build list of scene video paths
        scene_paths = [
            self.session_dir / scene["video_path"]
            for scene in scene_videos
        ]

        # Output path
        master_path = self.edit_output_dir / "master_final.mp4"

        # Concatenate with fade transitions between scenes
        self.ffmpeg_builder.concatenate_simple(
            scene_paths,
            master_path,
            add_fade_transitions=True,
            fade_duration=self.scene_fade_duration
        )

        logger.info(f"{self.agent_name}: Master timeline assembled: {master_path}")
        return master_path

    def _generate_heuristic_edl(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate simple heuristic-based EDL when Gemini fails.

        Rules:
        - Trim 0.5s from start/end if dialogue exists
        - If no dialogue, use speech boundaries with 0.3s padding
        - No J/L cuts (all hard cuts)
        - Simple concatenation

        Args:
            videos: List of video metadata
            audio_metadata: Audio analysis results
            scene_breakdown: Scene structure
            shot_breakdown: Shot details

        Returns:
            Heuristic EDL dictionary
        """
        logger.info(f"{self.agent_name}: Generating heuristic EDL")

        # Group shots by scene
        shots_by_scene = {}
        for shot in shot_breakdown.get("shots", []):
            scene_id = shot.get("scene_id")
            if scene_id not in shots_by_scene:
                shots_by_scene[scene_id] = []
            shots_by_scene[scene_id].append(shot)

        scenes = []
        for scene in scene_breakdown.get("scenes", []):
            scene_id = scene.get("scene_id")
            scene_shots = shots_by_scene.get(scene_id, [])

            shots_edl = []
            for i, shot in enumerate(scene_shots):
                shot_id = shot.get("shot_id")
                video_info = next((v for v in videos if v.get("shot_id") == shot_id), None)
                audio_info = audio_metadata.get(shot_id, {})

                if not video_info:
                    continue

                duration = video_info.get("duration_seconds", 5.0)
                dialogue = audio_info.get("dialogue")

                # Determine trim values with dialogue protection
                if dialogue and dialogue.get("has_speech"):
                    # Trim based on speech boundaries
                    trim_start = max(0.0, dialogue["speech_start"] - 0.3)
                    trim_end = min(duration, dialogue["speech_end"] + 0.3)

                    # CRITICAL: Never cut off dialogue
                    if trim_end < dialogue["speech_end"]:
                        logger.warning(
                            f"{shot_id}: trim_end ({trim_end}) would cut off dialogue at "
                            f"{dialogue['speech_end']}s - extending to protect speech"
                        )
                        trim_end = min(duration, dialogue["speech_end"] + 0.2)
                else:
                    # No speech, trim fixed amount
                    trim_start = 0.5
                    trim_end = duration - 0.5

                # Enforce maximum shot duration from config
                max_duration = self.config.get("max_shot_duration", 12.0)
                if trim_end - trim_start > max_duration:
                    logger.warning(f"{shot_id}: Shot exceeds max duration, clamping to {max_duration}s")
                    trim_end = trim_start + max_duration

                # Ensure minimum duration
                min_duration = self.config.get("min_shot_duration", 1.5)
                if trim_end - trim_start < min_duration:
                    # Try extending end first
                    trim_end = min(duration, trim_start + min_duration)
                    # If still too short, reset to use full shot
                    if trim_end - trim_start < min_duration:
                        trim_start = 0.0
                        trim_end = min(duration, min_duration)

                # FINAL VALIDATION: Never exceed original video duration
                if trim_end > duration:
                    logger.warning(f"{shot_id}: trim_end ({trim_end}) exceeds duration ({duration}), clamping")
                    trim_end = duration

                shots_edl.append({
                    "shot_id": shot_id,
                    "video_path": video_info.get("video_path"),
                    "edit_type": "hard_start" if i == 0 else "hard_cut",
                    "trim_start": round(trim_start, 2),
                    "trim_end": round(trim_end, 2),
                    "audio_start_offset": 0.0,
                    "transition": "cut",
                    "rationale": "Heuristic trim"
                })

            scenes.append({
                "scene_id": scene_id,
                "shots": shots_edl
            })

        return {
            "edit_plan": {
                "scenes": scenes,
                "editing_notes": "Heuristic fallback editing applied"
            },
            "editing_method": "heuristic_fallback"
        }
