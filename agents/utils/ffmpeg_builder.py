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
        trim_end: Optional[float] = None
    ) -> Path:
        """
        Edit a single shot: simple trim only.

        J/L cuts are NOT applied here - they're handled during concatenation.
        This keeps clips clean and synchronized.

        Args:
            video_path: Input video path
            output_path: Output video path
            trim_start: Seconds to trim from start
            trim_end: End timestamp (None = keep to end)

        Returns:
            Path to edited video
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Trimming shot: {video_path.name} ({trim_start}s to {trim_end if trim_end else 'end'}s)")

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

        # Simple trim command (video and audio stay synchronized)
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

        self._execute_ffmpeg(cmd, f"trimming shot {video_path.name}")

        logger.info(f"Shot trimmed: {output_path}")
        return output_path

    def concatenate_with_timeline_audio(
        self,
        video_files: List[Path],
        transitions: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Concatenate videos with proper J-cut and L-cut support using timeline-based audio mixing.

        J-cuts and L-cuts are cross-clip transitions that require audio from one shot
        to overlap with video from another shot. This method uses FFmpeg's adelay and
        amix filters to properly position and mix audio on a single timeline.

        Args:
            video_files: List of trimmed video file paths
            transitions: List of transition dicts (length = len(video_files) - 1):
                [
                    {"type": "j_cut", "audio_advance": 0.5},  # Shot 2 audio starts 0.5s before video cut
                    {"type": "l_cut", "audio_extend": 0.7},   # Shot 2 audio continues 0.7s after video cut
                    {"type": "hard_cut"}                       # No overlap
                ]
            output_path: Output video path

        Returns:
            Path to output video
        """
        if not video_files:
            raise ValueError("No video files provided")

        logger.info(f"Concatenating {len(video_files)} clips with {len(transitions)} transitions...")

        # Special case: Single video, just copy it
        if len(video_files) == 1:
            import shutil
            shutil.copy2(video_files[0], output_path)
            return output_path

        # Get durations of all clips
        durations = [self.get_video_duration(vf) for vf in video_files]

        # Calculate timeline positions for each clip
        # Account for J-cuts (audio starts early) and L-cuts (audio extends late)
        video_positions = [0.0]  # Video always starts at 0
        audio_positions = [0.0]  # Audio might start earlier due to J-cuts

        current_video_time = 0.0
        current_audio_time = 0.0

        for i, duration in enumerate(durations):
            if i < len(transitions):
                trans = transitions[i]
                trans_type = trans.get("type", "hard_cut")

                if trans_type == "j_cut":
                    # Next clip's audio starts early
                    audio_advance = trans.get("audio_advance", 0.0)
                    next_video_pos = current_video_time + duration
                    next_audio_pos = next_video_pos - audio_advance  # Start audio earlier

                    video_positions.append(next_video_pos)
                    audio_positions.append(next_audio_pos)
                    current_video_time = next_video_pos
                    current_audio_time = next_audio_pos

                elif trans_type == "l_cut":
                    # Current clip's audio extends beyond its video
                    audio_extend = trans.get("audio_extend", 0.0)
                    next_video_pos = current_video_time + duration
                    next_audio_pos = current_audio_time + duration  # Audio continues from current position

                    # Note: audio from current clip extends by audio_extend
                    # Next clip's audio will overlap

                    video_positions.append(next_video_pos)
                    audio_positions.append(next_video_pos - audio_extend)  # Audio starts to allow overlap
                    current_video_time = next_video_pos
                    current_audio_time = next_video_pos - audio_extend

                else:  # hard_cut
                    next_video_pos = current_video_time + duration
                    next_audio_pos = current_audio_time + duration

                    video_positions.append(next_video_pos)
                    audio_positions.append(next_audio_pos)
                    current_video_time = next_video_pos
                    current_audio_time = next_audio_pos

        # Validate audio positions to prevent sync issues
        logger.debug(f"Video positions: {video_positions}")
        logger.debug(f"Audio positions: {audio_positions}")

        for i in range(len(audio_positions)):
            # Get transition type for this clip
            if i == 0:
                trans_type = "hard_start"
            elif i - 1 < len(transitions):
                trans_type = transitions[i - 1].get("type", "hard_cut")
            else:
                trans_type = "hard_cut"

            # For hard cuts, audio must not start before video
            if trans_type in ["hard_cut", "hard_start"]:
                if audio_positions[i] < video_positions[i]:
                    logger.warning(
                        f"Clip {i}: Audio position ({audio_positions[i]:.2f}s) before video "
                        f"({video_positions[i]:.2f}s) - correcting to match video position"
                    )
                    audio_positions[i] = video_positions[i]

        # Optimization: If no J/L cuts, use simple concat (faster and guaranteed sync)
        has_jl_cuts = any(t.get("type") in ["j_cut", "l_cut"] for t in transitions)

        if not has_jl_cuts:
            logger.info("No J/L cuts detected - using simple concatenation for guaranteed sync")
            return self.concatenate_simple(video_files, output_path)

        logger.info(f"Processing {sum(1 for t in transitions if t.get('type') in ['j_cut', 'l_cut'])} J/L cuts...")

        # Build FFmpeg filter_complex
        input_args = []
        for vf in video_files:
            input_args.extend(["-i", str(vf)])

        # Build filter parts
        filter_parts = []

        # Concat video (simple, sequential)
        video_concat = "".join([f"[{i}:v]" for i in range(len(video_files))])
        video_concat += f"concat=n={len(video_files)}:v=1:a=0[vout]"
        filter_parts.append(video_concat)

        # Position and mix audio streams
        audio_labels = []
        for i in range(len(video_files)):
            audio_delay_ms = int(audio_positions[i] * 1000)

            if audio_delay_ms > 0:
                # Delay audio to correct timeline position
                filter_parts.append(f"[{i}:a]adelay={audio_delay_ms}|{audio_delay_ms}[a{i}]")
                audio_labels.append(f"[a{i}]")
            else:
                # No delay needed
                audio_labels.append(f"[{i}:a]")

        # Mix all audio streams (overlaps handled automatically)
        audio_mix = "".join(audio_labels)
        audio_mix += f"amix=inputs={len(video_files)}:duration=longest:dropout_transition=0[amixed]"
        filter_parts.append(audio_mix)

        # Build audio enhancement chain
        audio_enhancements = []
        current_label = "[amixed]"

        # [A] Dynamic Range Compression (make quiet parts louder, loud parts quieter)
        if self.config.get("audio_compression", True):
            compression_ratio = self.config.get("compression_ratio", 3)
            audio_enhancements.append(
                f"{current_label}acompressor=threshold=-18dB:ratio={compression_ratio}:attack=20:release=250[acomp]"
            )
            current_label = "[acomp]"
            logger.debug("Audio enhancement: Dynamic range compression enabled")

        # [C] Dialogue Enhancement EQ (boost voice frequencies)
        if self.config.get("dialogue_enhancement", True):
            eq_boost = self.config.get("eq_voice_boost_db", 3)
            audio_enhancements.append(
                f"{current_label}equalizer=f=1000:t=h:width=1000:g={eq_boost},highpass=f=100,lowpass=f=8000[aeq]"
            )
            current_label = "[aeq]"
            logger.debug(f"Audio enhancement: Dialogue EQ with +{eq_boost}dB boost")

        # [A] Loudness Normalization (streaming standard)
        if self.config.get("normalize_loudness", True):
            target_loudness = self.config.get("target_loudness", -16)
            audio_enhancements.append(
                f"{current_label}loudnorm=I={target_loudness}:TP=-1.5:LRA=11[aout]"
            )
            current_label = "[aout]"
            logger.debug(f"Audio enhancement: Loudness normalization to {target_loudness} LUFS")
        else:
            # Rename final label to aout
            if current_label != "[amixed]":
                audio_enhancements.append(f"{current_label}acopy[aout]")

        # Add all audio enhancements to filter
        if audio_enhancements:
            filter_parts.extend(audio_enhancements)

        filter_complex = ";".join(filter_parts)

        # Build final command
        cmd = [
            self.ffmpeg_path,
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", self.output_codec,
            "-preset", self.output_preset,
            "-crf", str(self.output_crf),
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-y",
            str(output_path)
        ]

        logger.debug(f"FFmpeg filter_complex: {filter_complex}")
        self._execute_ffmpeg(cmd, f"concatenation with audio mixing")

        logger.info(f"Timeline concatenation complete: {output_path}")
        return output_path

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
            # Step 1: Trim each shot (NO audio offsets applied yet)
            for i, shot_edit in enumerate(edl_shots):
                shot_id = shot_edit.get("shot_id")
                video_rel_path = shot_edit.get("video_path")
                video_path = session_dir / video_rel_path

                trim_start = shot_edit.get("trim_start", 0.0)
                trim_end = shot_edit.get("trim_end")

                # Output path for trimmed shot
                edited_shot_path = temp_dir / f"edited_{i:03d}_{shot_id}.mp4"

                # Trim the shot (audio stays synchronized with video)
                self.edit_shot(
                    video_path,
                    edited_shot_path,
                    trim_start,
                    trim_end
                )

                edited_shots.append(edited_shot_path)

            # Step 2: Build transitions list from EDL
            transitions = []
            for i in range(len(edl_shots) - 1):
                current_shot = edl_shots[i]
                next_shot = edl_shots[i + 1]

                edit_type = next_shot.get("edit_type", "hard_cut")
                audio_offset = next_shot.get("audio_start_offset", 0.0)

                if edit_type == "j_cut" and audio_offset < 0:
                    # J-cut: next shot's audio starts early
                    transitions.append({
                        "type": "j_cut",
                        "audio_advance": abs(audio_offset)
                    })
                    logger.debug(f"Transition {i}->{i+1}: J-cut with {abs(audio_offset)}s advance")

                elif edit_type == "l_cut" and audio_offset > 0:
                    # L-cut: current shot's audio extends
                    transitions.append({
                        "type": "l_cut",
                        "audio_extend": abs(audio_offset)
                    })
                    logger.debug(f"Transition {i}->{i+1}: L-cut with {abs(audio_offset)}s extension")

                else:
                    # Hard cut (or invalid offset)
                    transitions.append({"type": "hard_cut"})
                    logger.debug(f"Transition {i}->{i+1}: Hard cut")

            # Step 3: Concatenate with timeline-based audio mixing
            self.concatenate_with_timeline_audio(edited_shots, transitions, output_path)

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
