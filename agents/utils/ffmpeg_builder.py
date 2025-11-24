"""
FFmpeg Command Builder for Agent 11

Constructs complex FFmpeg filter graphs for:
- Video trimming
- Audio mixing with offsets (J-cuts, L-cuts)
- Concatenation with transitions
- Scene assembly
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


class FFmpegBuilder:
    """
    Builds and executes FFmpeg commands for intelligent video editing.

    Handles:
    - Trimming shots based on EDL
    - Audio offset for cinematic J/L cuts
    - Concatenation with transitions
    - Master timeline assembly
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FFmpegBuilder with configuration.

        Args:
            config: Agent 11 configuration dictionary
        """
        self.config = config
        self.ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        self.ffprobe_path = config.get("ffprobe_path", "ffprobe")

        # Encoding settings
        self.output_codec = config.get("output_codec", "libx264")
        self.output_preset = config.get("output_preset", "medium")
        self.output_crf = config.get("output_crf", 23)
        self.audio_codec = config.get("audio_codec", "aac")
        self.audio_bitrate = config.get("audio_bitrate", "192k")

        # Output settings
        self.output_resolution = config.get("output_resolution", "1920x1080")
        self.output_fps = config.get("output_fps", 24)

        # Verify FFmpeg installation
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify FFmpeg and FFprobe are installed and accessible."""
        for tool, path in [("FFmpeg", self.ffmpeg_path), ("FFprobe", self.ffprobe_path)]:
            try:
                result = subprocess.run(
                    [path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"{tool} verified at: {path}")
                else:
                    raise RuntimeError(f"{tool} failed verification")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"{tool} not found at '{path}'. "
                    f"Please install FFmpeg and update config.yaml. Error: {e}"
                )

    def _get_output_params(self, reference_video: Path) -> Dict[str, Any]:
        """
        Get output encoding parameters, auto-detecting from reference video if needed.

        Args:
            reference_video: Path to reference video for auto-detection

        Returns:
            Dictionary with resolution, fps, etc.
        """
        params = {}

        # Get video info if we need to auto-detect anything
        if (self.output_resolution == "auto" or
            self.output_fps == "auto" or
            isinstance(self.output_fps, str) and self.output_fps.lower() == "auto"):

            video_info = self.get_video_info(reference_video)
            video_stream = next(
                (s for s in video_info.get('streams', []) if s.get('codec_type') == 'video'),
                None
            )

            if not video_stream:
                logger.warning("No video stream found, using default values")
                params['width'] = 1920
                params['height'] = 1080
                params['fps'] = 24
            else:
                # Resolution
                if self.output_resolution == "auto":
                    params['width'] = video_stream.get('width', 1920)
                    params['height'] = video_stream.get('height', 1080)
                    logger.info(f"Auto-detected resolution: {params['width']}x{params['height']}")
                else:
                    # Parse resolution string like "1920x1080"
                    w, h = self.output_resolution.split('x')
                    params['width'] = int(w)
                    params['height'] = int(h)

                # FPS
                if isinstance(self.output_fps, str) and self.output_fps.lower() == "auto":
                    # Parse frame rate fraction (e.g., "24/1" or "30000/1001")
                    fps_str = video_stream.get('r_frame_rate', '24/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        params['fps'] = int(float(num) / float(den))
                    else:
                        params['fps'] = int(float(fps_str))
                    logger.info(f"Auto-detected fps: {params['fps']}")
                else:
                    params['fps'] = self.output_fps
        else:
            # Use configured values
            if self.output_resolution != "auto":
                w, h = self.output_resolution.split('x')
                params['width'] = int(w)
                params['height'] = int(h)
            else:
                params['width'] = 1920
                params['height'] = 1080

            params['fps'] = self.output_fps if self.output_fps != "auto" else 24

        return params

    def concatenate_simple(
        self,
        video_files: List[Path],
        output_path: Path,
        add_fade_transitions: bool = False,
        fade_duration: float = 1.0
    ) -> Path:
        """
        Simple concatenation of video files using concat demuxer.

        Args:
            video_files: List of video file paths to concatenate
            output_path: Output video file path
            add_fade_transitions: Whether to add fade transitions between videos
            fade_duration: Fade duration in seconds

        Returns:
            Path to output video
        """
        if not video_files:
            raise ValueError("No video files provided for concatenation")

        logger.info(f"Concatenating {len(video_files)} videos...")

        # Detect output parameters from first video if using "auto"
        output_params = self._get_output_params(video_files[0])
        logger.info(f"Output settings: {output_params['width']}x{output_params['height']} @ {output_params['fps']}fps")

        # Create concat file
        concat_file = output_path.parent / f"{output_path.stem}_concat.txt"

        with open(concat_file, 'w') as f:
            for video in video_files:
                if not video.exists():
                    raise FileNotFoundError(f"Video not found: {video}")
                # Use absolute paths and escape special characters
                f.write(f"file '{str(video.absolute())}'\n")

        try:
            if add_fade_transitions:
                # Use filter_complex for fade transitions
                output = self._concatenate_with_fades(
                    video_files,
                    output_path,
                    fade_duration
                )
            else:
                # Simple concat demuxer (faster)
                cmd = [
                    self.ffmpeg_path,
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c:v", self.output_codec,
                    "-preset", self.output_preset,
                    "-crf", str(self.output_crf),
                    "-c:a", self.audio_codec,
                    "-b:a", self.audio_bitrate,
                    "-y",  # Overwrite
                    str(output_path)
                ]

                self._execute_ffmpeg(cmd, f"concatenation of {len(video_files)} videos")
                output = output_path

            # Clean up concat file
            concat_file.unlink()

            logger.info(f"Concatenation complete: {output}")
            return output

        except Exception as e:
            # Clean up concat file on error
            if concat_file.exists():
                concat_file.unlink()
            raise

    def _concatenate_with_fades(
        self,
        video_files: List[Path],
        output_path: Path,
        fade_duration: float
    ) -> Path:
        """
        Concatenate videos with crossfade transitions.

        Args:
            video_files: List of video file paths
            output_path: Output video file path
            fade_duration: Fade duration in seconds

        Returns:
            Path to output video
        """
        if len(video_files) == 1:
            # Only one video, just copy it
            import shutil
            shutil.copy2(video_files[0], output_path)
            return output_path

        # Build filter_complex for xfade
        filter_parts = []
        input_args = []

        # Add all inputs
        for i, video in enumerate(video_files):
            input_args.extend(["-i", str(video)])

        # Build xfade chain
        # For n videos, we need n-1 xfades
        for i in range(len(video_files) - 1):
            if i == 0:
                # First xfade: [0][1] -> [v01]
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset=0[v01]"
                )
            else:
                # Subsequent xfades: [v0{i}][{i+1}] -> [v0{i+1}]
                prev_label = f"v0{i}" if i == 1 else f"v0{i}"
                curr_label = f"v0{i+1}"
                filter_parts.append(
                    f"[{prev_label}][{i+1}:v]xfade=transition=fade:duration={fade_duration}:offset=0[{curr_label}]"
                )

        # Audio mixing
        audio_filter = f"{''.join([f'[{i}:a]' for i in range(len(video_files))])}concat=n={len(video_files)}:v=0:a=1[aout]"
        filter_parts.append(audio_filter)

        filter_complex = ";".join(filter_parts)
        final_video_label = f"v0{len(video_files)-1}"

        cmd = [
            self.ffmpeg_path,
            *input_args,
            "-filter_complex", filter_complex,
            "-map", f"[{final_video_label}]",
            "-map", "[aout]",
            "-c:v", self.output_codec,
            "-preset", self.output_preset,
            "-crf", str(self.output_crf),
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-y",
            str(output_path)
        ]

        self._execute_ffmpeg(cmd, f"crossfade concatenation of {len(video_files)} videos")
        return output_path

    def edit_shot(
        self,
        video_path: Path,
        output_path: Path,
        trim_start: float = 0.0,
        trim_end: Optional[float] = None,
        audio_offset: float = 0.0
    ) -> Path:
        """
        Edit a single shot: trim and optionally offset audio.

        Args:
            video_path: Input video path
            output_path: Output video path
            trim_start: Seconds to trim from start
            trim_end: End timestamp (None = keep to end)
            audio_offset: Audio offset in seconds (negative = start early, positive = start late)

        Returns:
            Path to edited video
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Editing shot: {video_path.name}")

        # Get video duration
        duration = self.get_video_duration(video_path)

        # Calculate trim parameters
        if trim_end is None:
            trim_end = duration

        # Validate trim values
        trim_start = max(0.0, trim_start)
        trim_end = min(duration, trim_end)

        if trim_start >= trim_end:
            raise ValueError(f"Invalid trim values: start={trim_start}, end={trim_end}")

        # Build FFmpeg command
        if abs(audio_offset) < 0.01:
            # No audio offset, simple trim
            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-ss", str(trim_start),
                "-to", str(trim_end),
                "-c:v", self.output_codec,
                "-preset", self.output_preset,
                "-crf", str(self.output_crf),
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-y",
                str(output_path)
            ]
        else:
            # Audio offset required (for J/L cuts)
            filter_complex = self._build_audio_offset_filter(
                trim_start,
                trim_end,
                audio_offset,
                duration
            )

            cmd = [
                self.ffmpeg_path,
                "-i", str(video_path),
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", self.output_codec,
                "-preset", self.output_preset,
                "-crf", str(self.output_crf),
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-y",
                str(output_path)
            ]

        self._execute_ffmpeg(cmd, f"editing shot {video_path.name}")

        logger.info(f"Shot edited: {output_path}")
        return output_path

    def _build_audio_offset_filter(
        self,
        trim_start: float,
        trim_end: float,
        audio_offset: float,
        original_duration: float
    ) -> str:
        """
        Build filter_complex for audio offset (J/L cuts).

        Args:
            trim_start: Video trim start time
            trim_end: Video trim end time
            audio_offset: Audio offset in seconds
            original_duration: Original video duration

        Returns:
            FFmpeg filter_complex string
        """
        # Video trim
        video_filter = f"[0:v]trim=start={trim_start}:end={trim_end},setpts=PTS-STARTPTS[v]"

        # Audio trim with offset
        if audio_offset < 0:
            # J-cut: audio starts earlier
            audio_start = max(0, trim_start + audio_offset)
            audio_end = trim_end + audio_offset
            audio_filter = f"[0:a]atrim=start={audio_start}:end={audio_end},asetpts=PTS-STARTPTS,adelay={abs(audio_offset * 1000)}|{abs(audio_offset * 1000)}[a]"
        else:
            # L-cut: audio starts later
            audio_start = trim_start
            audio_end = min(original_duration, trim_end + audio_offset)
            audio_filter = f"[0:a]atrim=start={audio_start}:end={audio_end},asetpts=PTS-STARTPTS[a]"

        return f"{video_filter};{audio_filter}"

    def edit_scene_with_edl(
        self,
        edl_shots: List[Dict[str, Any]],
        session_dir: Path,
        output_path: Path
    ) -> Path:
        """
        Edit a scene using an Edit Decision List (EDL).

        Args:
            edl_shots: List of shot edit instructions from EDL
            session_dir: Session directory for resolving paths
            output_path: Output scene video path

        Returns:
            Path to edited scene video
        """
        logger.info(f"Editing scene with {len(edl_shots)} shots...")

        # Create temp directory for edited shots
        temp_dir = output_path.parent / "temp_edited_shots"
        temp_dir.mkdir(exist_ok=True)

        edited_shots = []

        try:
            # Edit each shot according to EDL
            for i, shot_edit in enumerate(edl_shots):
                shot_id = shot_edit.get("shot_id")
                video_rel_path = shot_edit.get("video_path")
                video_path = session_dir / video_rel_path

                trim_start = shot_edit.get("trim_start", 0.0)
                trim_end = shot_edit.get("trim_end")
                audio_offset = shot_edit.get("audio_start_offset", 0.0)

                # Output path for edited shot
                edited_shot_path = temp_dir / f"edited_{i:03d}_{shot_id}.mp4"

                # Edit the shot
                self.edit_shot(
                    video_path,
                    edited_shot_path,
                    trim_start,
                    trim_end,
                    audio_offset
                )

                edited_shots.append(edited_shot_path)

            # Concatenate edited shots
            self.concatenate_simple(edited_shots, output_path)

            # Clean up temp files
            for shot in edited_shots:
                shot.unlink()
            temp_dir.rmdir()

            logger.info(f"Scene editing complete: {output_path}")
            return output_path

        except Exception as e:
            # Clean up on error
            for shot in edited_shots:
                if shot.exists():
                    shot.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
            raise

    def get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration using FFprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        return float(result.stdout.strip())

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive video information using FFprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration",
            "-show_entries", "format=duration,size,bit_rate",
            "-of", "json",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        import json
        return json.loads(result.stdout)

    def _execute_ffmpeg(self, cmd: List[str], operation_desc: str):
        """
        Execute FFmpeg command with error handling.

        Args:
            cmd: FFmpeg command as list
            operation_desc: Description of operation for logging
        """
        logger.debug(f"Executing FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed during {operation_desc}")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        logger.debug(f"FFmpeg {operation_desc} completed successfully")
