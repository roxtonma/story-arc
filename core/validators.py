"""
Data Validators
Pydantic models for validating data structures across the pipeline.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


# ==================== Agent 2: Scene Breakdown ====================

class CharacterDescription(BaseModel):
    """Character description with full details."""
    name: str = Field(..., description="Character name")
    description: str = Field(..., description="Painfully verbose physical appearance")


class LocationDescription(BaseModel):
    """Location description with full details."""
    name: str = Field(..., description="Location name")
    description: str = Field(..., description="Painfully verbose description including objects and positions")


class Subscene(BaseModel):
    """Subscene within a scene (for character additions without location change)."""
    subscene_id: str = Field(..., description="Subscene identifier (e.g., '1.1', '1.2')")
    event: Literal["CHARACTER_ADDED", "DIALOGUE", "ACTION"] = Field(
        ..., description="Type of subscene event"
    )
    character_added: Optional[CharacterDescription] = Field(
        None, description="Character details if CHARACTER_ADDED event"
    )
    dialogue: Optional[str] = Field(None, description="Dialogue content")
    action: Optional[str] = Field(None, description="Action description")
    screenplay_excerpt: str = Field(..., description="Relevant screenplay text for this subscene")


class Scene(BaseModel):
    """Scene with location, characters, and subscenes."""
    scene_id: str = Field(..., description="Scene identifier (e.g., 'SCENE_1')")
    location: LocationDescription = Field(..., description="Location details")
    characters: List[CharacterDescription] = Field(
        ..., description="All characters present in scene"
    )
    time_of_day: Optional[str] = Field(None, description="Time of day (e.g., 'DAY', 'NIGHT')")
    screenplay_text: str = Field(..., description="Full screenplay text for this scene")
    subscenes: List[Subscene] = Field(
        default_factory=list,
        description="Subscenes for character additions and major events"
    )


class SceneBreakdown(BaseModel):
    """Complete scene breakdown from Agent 2."""
    scenes: List[Scene] = Field(..., description="List of all scenes")
    total_scenes: int = Field(..., description="Total number of scenes")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ==================== Agent 3: Shot Breakdown ====================

class Shot(BaseModel):
    """Individual shot with description, first frame, and animation."""
    shot_id: str = Field(..., description="Shot identifier (e.g., 'SHOT_1_1')")
    scene_id: str = Field(..., description="Parent scene identifier")
    shot_description: str = Field(
        ...,
        description="Detailed description of what happens in the shot"
    )
    first_frame: str = Field(
        ...,
        description="Painfully verbose description of the first frame with ALL objects/people present"
    )
    animation: str = Field(
        ...,
        description="Description of what the video model should do to convert first frame into the shot"
    )
    characters: List[str] = Field(
        default_factory=list,
        description="List of character names in this shot"
    )
    location: str = Field(..., description="Location name for this shot")
    dialogue: Optional[str] = Field(
        None,
        description="Dialogue spoken in this shot (guideline: 6-7 words, can create multiple shots for longer dialogue)"
    )

    @validator("first_frame")
    def validate_first_frame(cls, v):
        """Ensure first frame description is detailed."""
        if len(v) < 10:
            raise ValueError("First frame description must be painfully verbose (at least 10 characters)")
        return v


class ShotBreakdown(BaseModel):
    """Complete shot breakdown from Agent 3."""
    shots: List[Shot] = Field(..., description="List of all shots")
    total_shots: int = Field(..., description="Total number of shots")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ==================== Agent 4: Shot Grouping ====================

class GroupedShot(BaseModel):
    """Shot with parent/child relationships (reference-based for scalability)."""
    shot_id: str = Field(..., description="Shot identifier (references shot from Agent 3)")
    scene_id: str = Field(..., description="Scene identifier")
    shot_type: Literal["parent", "child"] = Field(..., description="Whether this is a parent or child shot")
    parent_shot_id: Optional[str] = Field(
        None,
        description="Parent shot ID if this is a child shot"
    )
    child_shots: List["GroupedShot"] = Field(
        default_factory=list,
        description="List of child shots if this is a parent shot"
    )
    grouping_reason: str = Field(
        ...,
        description="Reason for grouping (e.g., 'same location and characters', 'character addition')"
    )


# Enable forward references for recursive model
GroupedShot.model_rebuild()


class ShotGrouping(BaseModel):
    """Complete shot grouping from Agent 4."""
    parent_shots: List[GroupedShot] = Field(
        ...,
        description="Top-level parent shots with nested children"
    )
    total_parent_shots: int = Field(..., description="Total number of parent shots")
    total_child_shots: int = Field(..., description="Total number of child shots")
    grouping_strategy: str = Field(
        ...,
        description="Description of grouping strategy used"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ==================== Phase 2: Image Generation ====================

class CharacterData(BaseModel):
    """Generated character image data."""
    name: str = Field(..., description="Character name")
    description: str = Field(..., description="Visual description used for generation")
    image_path: str = Field(..., description="Relative path to character image")
    generation_timestamp: str = Field(..., description="When the image was generated")


class CharacterGrid(BaseModel):
    """Character combination grid for shot reference."""
    grid_id: str = Field(..., description="Unique grid identifier")
    characters: List[str] = Field(..., description="Character names in this grid")
    grid_path: str = Field(..., description="Relative path to grid image")
    generation_timestamp: str = Field(..., description="When the grid was generated")


class CharacterCreationOutput(BaseModel):
    """Complete output from Agent 5."""
    characters: List[CharacterData] = Field(..., description="All generated characters")
    character_grids: List[CharacterGrid] = Field(..., description="Character combination grids")
    total_characters: int = Field(..., description="Total number of characters")
    total_grids: int = Field(..., description="Total number of grids created")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class VerificationIssue(BaseModel):
    """Single verification issue with category."""
    category: str = Field(..., description="Issue category (e.g., 'Extra Character', 'Grid Artifact')")
    description: str = Field(..., description="Detailed description of the issue")


class VerificationResult(BaseModel):
    """Image verification result."""
    approved: bool = Field(..., description="Whether image is approved")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    issues: List[VerificationIssue] = Field(
        default_factory=list,
        description="List of categorized issues found"
    )
    recommendation: Literal["approve", "regenerate", "manual_review"] = Field(
        ...,
        description="Recommended action"
    )


class GeneratedShotImage(BaseModel):
    """Generated image for a shot."""
    shot_id: str = Field(..., description="Shot identifier")
    scene_id: str = Field(..., description="Scene identifier")
    image_path: str = Field(..., description="Relative path to generated image")
    generation_timestamp: str = Field(..., description="Generation timestamp")
    verification_status: Literal["pending", "verified", "soft_failure"] = Field(
        ...,
        description="Verification status"
    )
    attempts: int = Field(default=1, description="Number of generation attempts")
    final_verification: Optional[VerificationResult] = Field(
        None,
        description="Final verification result"
    )
    verification_history: List[VerificationResult] = Field(
        default_factory=list,
        description="History of all verification attempts"
    )


class ParentShotsOutput(BaseModel):
    """Complete output from Agent 6."""
    parent_shots: List[GeneratedShotImage] = Field(..., description="All parent shot images")
    total_parent_shots: int = Field(..., description="Total number of parent shots")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ChildShotsOutput(BaseModel):
    """Complete output from Agent 8."""
    child_shots: List[GeneratedShotImage] = Field(..., description="All child shot images")
    total_child_shots: int = Field(..., description="Total number of child shots")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ==================== Agent 10: Video Dialogue Generation ====================

class TimeSegment(BaseModel):
    """Single time segment in video production timeline."""
    time_segment: str = Field(..., description="Time range (e.g., '0.0-2.0s')")
    visual_event: str = Field(..., description="What happens visually")
    audio_event: str = Field(..., description="Dialogue, voice-over, or silence")
    constraints_segment: List[str] = Field(..., description="Constraints for this segment")


class Cinematography(BaseModel):
    """Cinematography details for the shot."""
    shot_type: str = Field(..., description="Type of shot (close-up, medium, wide, etc.)")
    camera_movement: str = Field(..., description="Camera movement description")


class VideoProductionBrief(BaseModel):
    """Complete production brief for video generation."""
    title: str = Field(..., description="3-5 word shot title")
    duration_seconds: int = Field(..., ge=4, le=8, description="Video duration (4, 6, or 8 seconds)")
    objective: str = Field(..., description="Visual or emotional goal")
    art_style: str = Field(..., description="Visual and lighting style")
    temporal_action_plan: List[TimeSegment] = Field(..., description="Timeline of actions")
    cinematography: Cinematography = Field(..., description="Camera and shot details")
    atmosphere_and_tone: str = Field(..., description="Visual adjectives only")
    constraints: List[str] = Field(..., description="Production constraints")
    video_generation_prompt: str = Field(..., description="Single-line prompt for video API")


class ProductionBriefResponse(BaseModel):
    """Response wrapper for production brief."""
    video_production_brief: VideoProductionBrief


class VideoOutput(BaseModel):
    """Generated video data for a single shot."""
    shot_id: str = Field(..., description="Shot identifier")
    shot_type: Literal["parent", "child"] = Field(..., description="Shot type")
    image_path: str = Field(..., description="Path to source image")
    video_url: str = Field(..., description="URL of generated video from FAL")
    video_path: Optional[str] = Field(None, description="Path to downloaded video file")
    production_brief_path: str = Field(..., description="Path to production brief JSON")
    duration_seconds: int = Field(..., ge=4, le=8, description="Video duration (4, 6, or 8 seconds)")
    video_prompt: str = Field(..., description="Prompt used for video generation")
    fal_request_id: str = Field(..., description="FAL API request ID")
    generated_at: str = Field(..., description="Generation timestamp")
    status: Literal["success", "failed"] = Field(..., description="Generation status")
    error: Optional[str] = Field(None, description="Error message if failed")


class VideoDialogueOutput(BaseModel):
    """Complete output from Agent 10."""
    videos: List[VideoOutput] = Field(..., description="All generated videos")
    total_videos: int = Field(..., description="Total number of videos")
    successful_videos: int = Field(..., description="Number of successful videos")
    failed_videos: int = Field(..., description="Number of failed videos")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata including session_id, models used"
    )


# ==================== Agent 11: Video Editing ====================

class EditShot(BaseModel):
    """Edit instructions for a single shot in the Edit Decision List."""
    shot_id: str = Field(..., description="Shot identifier")
    video_path: str = Field(..., description="Relative path to video file")
    edit_type: Literal["hard_start", "j_cut", "l_cut", "hard_cut"] = Field(
        ...,
        description="Type of edit: hard_start (scene start), j_cut (audio leads), l_cut (audio lags), hard_cut (simple)"
    )
    trim_start: float = Field(..., ge=0.0, description="Seconds to trim from start of video")
    trim_end: float = Field(..., gt=0.0, description="End timestamp in source video (not duration)")
    audio_start_offset: float = Field(
        default=0.0,
        description="Audio offset in seconds: negative=J-cut (audio early), positive=L-cut (audio late), 0=sync"
    )
    transition: Literal["cut", "fade", "dissolve"] = Field(
        default="cut",
        description="Transition type (currently only 'cut' is implemented)"
    )
    rationale: str = Field(..., description="Brief explanation of edit decision")

    @validator("trim_end")
    def validate_trim_end_gt_start(cls, v, values):
        """Ensure trim_end is greater than trim_start."""
        if "trim_start" in values and v <= values["trim_start"]:
            raise ValueError("trim_end must be greater than trim_start")
        return v

    @validator("audio_start_offset")
    def validate_audio_offset_rules(cls, v, values):
        """Validate audio offset rules based on edit_type."""
        if "edit_type" in values:
            edit_type = values["edit_type"]
            if edit_type in ["hard_start", "hard_cut"] and abs(v) > 0.01:
                raise ValueError(f"{edit_type} must have audio_start_offset of 0.0")
            if edit_type == "j_cut" and v >= 0.0:
                raise ValueError("j_cut requires negative audio_start_offset")
            if edit_type == "l_cut" and v <= 0.0:
                raise ValueError("l_cut requires positive audio_start_offset")
        return v


class SceneEdit(BaseModel):
    """Edit instructions for a complete scene."""
    scene_id: str = Field(..., description="Scene identifier (e.g., 'SCENE_1')")
    shots: List[EditShot] = Field(..., description="Ordered list of shot edits for this scene")

    @validator("shots")
    def validate_first_shot_is_hard_start(cls, v):
        """Ensure first shot of scene uses hard_start."""
        if v and v[0].edit_type != "hard_start":
            raise ValueError(
                f"First shot of scene must use edit_type='hard_start', "
                f"got '{v[0].edit_type}'"
            )
        return v


class EditPlan(BaseModel):
    """Complete edit plan from Gemini."""
    scenes: List[SceneEdit] = Field(..., description="List of scene edit instructions")
    total_estimated_duration: Optional[float] = Field(
        None,
        description="Estimated total duration after editing (seconds)"
    )
    scene_count: int = Field(..., description="Number of scenes")
    editing_notes: Optional[str] = Field(
        None,
        description="General notes about editing approach"
    )


class EditTimeline(BaseModel):
    """Complete edit timeline/EDL structure."""
    edit_plan: EditPlan = Field(..., description="The edit plan from Gemini")
    editing_method: Literal["gemini_edl", "heuristic_fallback"] = Field(
        default="gemini_edl",
        description="Method used to generate EDL"
    )
    gemini_attempt: Optional[int] = Field(
        None,
        description="Gemini attempt number if using gemini_edl method"
    )


class SceneVideo(BaseModel):
    """Metadata for an edited scene video."""
    scene_id: str = Field(..., description="Scene identifier")
    video_path: str = Field(..., description="Relative path to edited scene video")
    shot_count: int = Field(..., ge=1, description="Number of shots in this scene")
    duration: float = Field(..., gt=0.0, description="Scene duration in seconds")


class VideoEditOutput(BaseModel):
    """Complete output from Agent 11."""
    master_video_path: str = Field(
        ...,
        description="Relative path to master video (all scenes combined)"
    )
    scene_videos: List[SceneVideo] = Field(
        ...,
        description="List of individual scene videos"
    )
    edit_timeline: EditTimeline = Field(
        ...,
        description="Complete edit decision list used"
    )
    total_duration: float = Field(
        ...,
        gt=0.0,
        description="Total duration of master video in seconds"
    )
    edit_metadata: Dict[str, Any] = Field(
        ...,
        description="Editing metadata (scenes_edited, total_shots, editing_method, etc.)"
    )


# ==================== Session Management ====================

class AgentOutput(BaseModel):
    """Output from a single agent."""
    agent_name: str = Field(..., description="Agent identifier")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        ...,
        description="Agent execution status"
    )
    output_data: Optional[Any] = Field(None, description="Agent output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: str = Field(..., description="ISO timestamp of execution")
    retry_count: int = Field(default=0, description="Number of retry attempts")


class SessionState(BaseModel):
    """Complete session state for resumability."""
    session_id: str = Field(..., description="Unique session identifier")
    session_name: Optional[str] = Field(None, description="User-friendly session name")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    start_agent: Literal["agent_1", "agent_2"] = Field(
        ...,
        description="Starting agent for this session"
    )
    current_agent: str = Field(..., description="Currently executing/last completed agent")
    agents: Dict[str, AgentOutput] = Field(
        ...,
        description="Dictionary of agent outputs keyed by agent name"
    )
    input_data: str = Field(..., description="Original input data")
    status: Literal["in_progress", "completed", "failed", "paused"] = Field(
        ...,
        description="Overall session status"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional session metadata"
    )
