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
        if len(v) < 50:
            raise ValueError("First frame description must be painfully verbose (at least 50 characters)")
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
