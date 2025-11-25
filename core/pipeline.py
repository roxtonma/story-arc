"""
Pipeline Orchestrator
Manages the execution of all agents and session state.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from loguru import logger

from core.gemini_client import GeminiClient
from core.session_manager import SessionManager
from core.validators import SessionState
from agents.agent_1_screenplay import ScreenplayAgent
from agents.agent_2_scene_breakdown import SceneBreakdownAgent
from agents.agent_3_shot_breakdown import ShotBreakdownAgent
from agents.agent_4_grouping import ShotGroupingAgent
# Phase 2: Image Generation Agents
from agents.agent_5_character import CharacterCreatorAgent
from agents.agent_6_parent_generator import ParentImageGeneratorAgent
from agents.agent_7_parent_verification import ParentVerificationAgent
from agents.agent_8_child_generator import ChildImageGeneratorAgent
from agents.agent_9_child_verification import ChildVerificationAgent
from agents.agent_10_video_dialogue import VideoDialogueAgent
# Phase 3: Video Editing
from agents.agent_11_video_edit import VideoEditAgent


class Pipeline:
    """Orchestrates the multi-agent pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize Gemini client with Vertex AI support
        gemini_config = self.config["gemini"]
        use_vertex_ai = gemini_config.get("use_vertex_ai", False)

        if use_vertex_ai:
            # Vertex AI mode
            vertex_config = gemini_config.get("vertex_ai", {})
            self.gemini_client = GeminiClient(
                model_name=gemini_config["model"],
                max_retries=self.config["app"]["max_retries"],
                use_vertex_ai=True,
                vertex_project=vertex_config.get("project_id"),
                vertex_location=vertex_config.get("location"),
                vertex_credentials_file=vertex_config.get("credentials_file")
            )
        else:
            # Direct API mode (existing behavior)
            self.gemini_client = GeminiClient(
                model_name=gemini_config["model"],
                max_retries=self.config["app"]["max_retries"]
            )

        # Initialize session manager
        self.session_manager = SessionManager(
            base_directory=self.config["output"]["base_directory"]
        )

        # Initialize agents
        self.agents = self._initialize_agents()

        # Agent execution order
        # Phase 1: Script to shot breakdown
        # Phase 2: Image generation (agents 5-9)
        # Phase 3: Video generation & editing (agents 10-11)
        self.agent_order = [
            "agent_1", "agent_2", "agent_3", "agent_4",  # Phase 1
            "agent_5", "agent_6", "agent_7", "agent_8", "agent_9",  # Phase 2
            "agent_10", "agent_11"  # Phase 3
        ]

        logger.info("Pipeline initialized successfully (Phase 1 + Phase 2 + Phase 3)")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded configuration from {config_path}")
        return config

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents."""
        agents = {}

        # Agent 1: Screenplay Generator
        if self.config["agents"]["agent_1"]["enabled"]:
            agents["agent_1"] = ScreenplayAgent(
                self.gemini_client,
                self.config["agents"]["agent_1"]
            )

        # Agent 2: Scene Breakdown
        if self.config["agents"]["agent_2"]["enabled"]:
            agents["agent_2"] = SceneBreakdownAgent(
                self.gemini_client,
                self.config["agents"]["agent_2"]
            )

        # Agent 3: Shot Breakdown
        if self.config["agents"]["agent_3"]["enabled"]:
            agents["agent_3"] = ShotBreakdownAgent(
                self.gemini_client,
                self.config["agents"]["agent_3"]
            )

        # Agent 4: Shot Grouping
        if self.config["agents"]["agent_4"]["enabled"]:
            agents["agent_4"] = ShotGroupingAgent(
                self.gemini_client,
                self.config["agents"]["agent_4"]
            )

        # Phase 2 agents are initialized per-session (need session_dir)
        # They're created on-demand in _get_phase2_agent()

        logger.info(f"Initialized {len(agents)} Phase 1 agents")
        return agents

    def _get_phase2_agent(self, agent_name: str, session: SessionState):
        """
        Initialize Phase 2/3 agent on-demand with session directory.

        Args:
            agent_name: Agent name (agent_5 through agent_11)
            session: Current session state

        Returns:
            Initialized Phase 2/3 agent
        """
        if agent_name not in self.agent_order:
            raise ValueError(f"Unknown agent: {agent_name}")

        if agent_name not in ["agent_5", "agent_6", "agent_7", "agent_8", "agent_9", "agent_10", "agent_11"]:
            raise ValueError(f"{agent_name} is not a Phase 2/3 agent")

        # Get session directory
        session_dir = Path(self.config["output"]["base_directory"]) / session.session_id

        # Initialize agent based on name
        if agent_name == "agent_5" and self.config["agents"]["agent_5"]["enabled"]:
            return CharacterCreatorAgent(
                self.gemini_client,
                self.config["agents"]["agent_5"],
                session_dir
            )

        elif agent_name == "agent_6" and self.config["agents"]["agent_6"]["enabled"]:
            return ParentImageGeneratorAgent(
                self.gemini_client,
                self.config["agents"]["agent_6"],
                session_dir
            )

        elif agent_name == "agent_7" and self.config["agents"]["agent_7"]["enabled"]:
            return ParentVerificationAgent(
                self.gemini_client,
                self.config["agents"]["agent_7"],
                session_dir
            )

        elif agent_name == "agent_8" and self.config["agents"]["agent_8"]["enabled"]:
            return ChildImageGeneratorAgent(
                self.gemini_client,
                self.config["agents"]["agent_8"],
                session_dir
            )

        elif agent_name == "agent_9" and self.config["agents"]["agent_9"]["enabled"]:
            return ChildVerificationAgent(
                self.gemini_client,
                self.config["agents"]["agent_9"],
                session_dir
            )

        elif agent_name == "agent_10" and self.config["agents"]["agent_10"]["enabled"]:
            return VideoDialogueAgent(
                self.gemini_client,
                self.config["agents"]["agent_10"],
                session_dir
            )

        elif agent_name == "agent_11" and self.config["agents"]["agent_11"]["enabled"]:
            return VideoEditAgent(
                self.gemini_client,
                self.config["agents"]["agent_11"],
                session_dir
            )

        else:
            raise ValueError(f"Agent {agent_name} is not enabled or not found")

    def create_session(
        self,
        input_data: str,
        start_agent: str = "agent_1",
        session_name: Optional[str] = None
    ) -> SessionState:
        """
        Create a new pipeline session.

        Args:
            input_data: Initial input data
            start_agent: Starting agent ('agent_1' or 'agent_2')
            session_name: Optional session name

        Returns:
            SessionState object
        """
        if start_agent not in ["agent_1", "agent_2"]:
            raise ValueError("start_agent must be 'agent_1' or 'agent_2'")

        session = self.session_manager.create_session(
            input_data=input_data,
            start_agent=start_agent,
            session_name=session_name
        )

        logger.info(f"Created session: {session.session_id}")
        return session

    def run_agent(
        self,
        session: SessionState,
        agent_name: str,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Run a single agent.

        Args:
            session: Current session state
            agent_name: Agent to run
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state
        """
        max_retries = self.config["app"]["max_retries"]

        # Get agent (Phase 1 from cache, Phase 2/3 created on-demand)
        if agent_name in ["agent_5", "agent_6", "agent_7", "agent_8", "agent_9", "agent_10", "agent_11"]:
            # Phase 2/3 agent - initialize with session directory
            agent = self._get_phase2_agent(agent_name, session)
        elif agent_name in self.agents:
            # Phase 1 agent - use cached instance
            agent = self.agents[agent_name]
        else:
            raise ValueError(f"Agent not found: {agent_name}")

        logger.info(f"Running {agent_name}...")

        if progress_callback:
            progress_callback(f"Running {agent_name}...", 0)

        # Get input data for this agent
        input_data = self._get_agent_input(session, agent_name)

        # Execute agent with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"{agent_name}: Attempt {attempt + 1}/{max_retries}")

                if progress_callback:
                    progress_callback(
                        f"Running {agent_name} (attempt {attempt + 1}/{max_retries})...",
                        (attempt + 1) / max_retries * 0.5
                    )

                output_data = agent.execute(input_data)

                # Update session with successful output
                session = self.session_manager.update_agent_output(
                    session=session,
                    agent_name=agent_name,
                    output_data=output_data,
                    status="completed",
                    retry_count=attempt
                )

                logger.info(f"{agent_name}: Completed successfully")

                if progress_callback:
                    progress_callback(f"{agent_name} completed successfully", 1.0)

                return session

            except Exception as e:
                logger.warning(f"{agent_name}: Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    logger.info(f"{agent_name}: Retrying...")
                    continue
                else:
                    # All attempts failed - update session with error
                    session = self.session_manager.update_agent_output(
                        session=session,
                        agent_name=agent_name,
                        output_data=None,
                        status="failed",
                        error_message=str(e),
                        retry_count=attempt + 1
                    )

                    logger.error(f"{agent_name}: Failed after {max_retries} attempts")

                    if progress_callback:
                        progress_callback(
                            f"{agent_name} failed: {str(e)}",
                            1.0,
                            error=True
                        )

                    raise Exception(
                        f"{agent_name} failed after {max_retries} attempts: {str(e)}"
                    )

    def run_pipeline(
        self,
        session: SessionState,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Run the complete pipeline from the session's current state.

        Args:
            session: Session state
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state with all agents completed
        """
        logger.info(f"Running pipeline for session: {session.session_id}")

        # Determine which agents to run
        start_idx = self.agent_order.index(session.start_agent)
        agents_to_run = self.agent_order[start_idx:]

        # Skip agents that are already completed
        for i, agent_name in enumerate(agents_to_run):
            if agent_name in session.agents and session.agents[agent_name].status == "completed":
                logger.info(f"Skipping {agent_name} (already completed)")
                continue

            # Run agent
            try:
                if progress_callback:
                    progress = (i / len(agents_to_run))
                    progress_callback(
                        f"Pipeline progress: {i}/{len(agents_to_run)} agents completed",
                        progress
                    )

                session = self.run_agent(session, agent_name, progress_callback)

            except Exception as e:
                logger.error(f"Pipeline failed at {agent_name}: {str(e)}")
                session.status = "failed"
                self.session_manager.save_session(session)
                raise

        # Mark pipeline as completed
        session.status = "completed"
        self.session_manager.save_session(session)

        logger.info(f"Pipeline completed successfully for session: {session.session_id}")

        if progress_callback:
            progress_callback("Pipeline completed successfully", 1.0)

        return session

    def _phase_1_complete(self, session: SessionState) -> bool:
        """Check if Phase 1 (Agents 1-4) is complete."""
        return all(
            f"agent_{i}" in session.agents and
            session.agents[f"agent_{i}"].status == "completed"
            for i in range(1, 5)
        )

    def _phase_2_complete(self, session: SessionState) -> bool:
        """Check if Phase 2 (Agents 5-9) is complete."""
        return all(
            f"agent_{i}" in session.agents and
            session.agents[f"agent_{i}"].status == "completed"
            for i in range(5, 10)
        )

    def _phase_3_complete(self, session: SessionState) -> bool:
        """Check if Phase 3 (Agent 10) is complete."""
        return "agent_10" in session.agents and \
               session.agents["agent_10"].status == "completed"

    def _run_phase(
        self,
        session: SessionState,
        agents: list,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Internal method to run a list of agents sequentially.

        Args:
            session: Session state
            agents: List of agent names to run
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state
        """
        for i, agent_name in enumerate(agents):
            # Skip already completed agents
            if agent_name in session.agents and session.agents[agent_name].status == "completed":
                logger.info(f"Skipping {agent_name} (already completed)")
                continue

            # Run agent
            try:
                if progress_callback:
                    progress = (i / len(agents))
                    progress_callback(
                        f"Phase progress: {i}/{len(agents)} agents completed",
                        progress
                    )

                session = self.run_agent(session, agent_name, progress_callback)

            except Exception as e:
                logger.error(f"Phase failed at {agent_name}: {str(e)}")
                session.status = "failed"
                self.session_manager.save_session(session)
                raise

        return session

    def run_phase_1(
        self,
        session: SessionState,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Run Phase 1: Script to Shot (Agents 1-4).

        Args:
            session: Session state
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state with Phase 1 completed
        """
        logger.info(f"Running Phase 1 for session: {session.session_id}")

        phase_agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
        session = self._run_phase(session, phase_agents, progress_callback)

        logger.info(f"Phase 1 completed for session: {session.session_id}")

        if progress_callback:
            progress_callback("Phase 1 completed successfully", 1.0)

        return session

    def run_phase_2(
        self,
        session: SessionState,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Run Phase 2: Image Generation (Agents 5-9).

        Requires Phase 1 to be completed first.

        Args:
            session: Session state
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state with Phase 2 completed

        Raises:
            ValueError: If Phase 1 is not complete
        """
        logger.info(f"Running Phase 2 for session: {session.session_id}")

        # Verify Phase 1 is complete
        if not self._phase_1_complete(session):
            raise ValueError("Phase 1 must be completed before running Phase 2")

        phase_agents = ["agent_5", "agent_6", "agent_7", "agent_8", "agent_9"]
        session = self._run_phase(session, phase_agents, progress_callback)

        logger.info(f"Phase 2 completed for session: {session.session_id}")

        if progress_callback:
            progress_callback("Phase 2 completed successfully", 1.0)

        return session

    def run_phase_3(
        self,
        session: SessionState,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Run Phase 3: Video Generation (Agent 10).

        Requires Phase 2 to be completed first.

        Args:
            session: Session state
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state with Phase 3 completed

        Raises:
            ValueError: If Phase 2 is not complete
        """
        logger.info(f"Running Phase 3 for session: {session.session_id}")

        # Verify Phase 2 is complete
        if not self._phase_2_complete(session):
            raise ValueError("Phase 2 must be completed before running Phase 3")

        phase_agents = ["agent_10"]
        session = self._run_phase(session, phase_agents, progress_callback)

        # Mark as completed if all phases done
        if self._phase_1_complete(session) and self._phase_2_complete(session) and self._phase_3_complete(session):
            session.status = "completed"
            self.session_manager.save_session(session)

        logger.info(f"Phase 3 completed for session: {session.session_id}")

        if progress_callback:
            progress_callback("Phase 3 completed successfully", 1.0)

        return session

    def resume_from_agent(
        self,
        session: SessionState,
        agent_name: str,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Resume pipeline execution from a specific agent.

        Args:
            session: Session state
            agent_name: Agent to resume from
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state
        """
        if not self.session_manager.can_resume_from_agent(session, agent_name):
            raise ValueError(
                f"Cannot resume from {agent_name}. "
                "Previous agents must be completed first."
            )

        logger.info(f"Resuming session {session.session_id} from {agent_name}")

        # Update current agent
        session.current_agent = agent_name
        session.status = "in_progress"

        # Run from this agent onwards
        start_idx = self.agent_order.index(agent_name)
        agents_to_run = self.agent_order[start_idx:]

        for agent in agents_to_run:
            session = self.run_agent(session, agent, progress_callback)

        session.status = "completed"
        self.session_manager.save_session(session)

        logger.info(f"Resumed pipeline completed for session: {session.session_id}")
        return session

    def retry_agent(
        self,
        session: SessionState,
        agent_name: str,
        modified_input: Optional[Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> SessionState:
        """
        Retry a failed or completed agent with optional modified input.

        Args:
            session: Session state
            agent_name: Agent to retry
            modified_input: Optional modified input data (if None, uses original input)
            progress_callback: Optional callback for progress updates

        Returns:
            Updated session state with retried agent

        Raises:
            ValueError: If agent doesn't exist or dependencies not met
        """
        logger.info(f"Retrying {agent_name} for session: {session.session_id}")

        # Validate agent exists
        if agent_name not in self.agent_order:
            raise ValueError(f"Invalid agent name: {agent_name}")

        # Reset agent status to pending
        if agent_name in session.agents:
            session.agents[agent_name].status = "pending"
            session.agents[agent_name].error_message = None
            session.agents[agent_name].retry_count = 0

        # Get input - use modified input if provided, otherwise use default
        if modified_input is not None:
            input_data = modified_input
            logger.info(f"Using modified input for {agent_name}")
        else:
            input_data = self._get_agent_input(session, agent_name)
            logger.info(f"Using original input for {agent_name}")

        # Save session before retry
        self.session_manager.save_session(session)

        # Run the agent
        try:
            if progress_callback:
                progress_callback(f"Retrying {agent_name}...", 0.0)

            session = self.run_agent(session, agent_name, progress_callback)

            if progress_callback:
                progress_callback(f"{agent_name} retry completed successfully", 1.0)

            logger.info(f"Successfully retried {agent_name}")

        except Exception as e:
            logger.error(f"Failed to retry {agent_name}: {str(e)}")
            if progress_callback:
                progress_callback(f"{agent_name} retry failed: {str(e)}", 1.0, error=True)
            raise

        return session

    def generate_single_shot_video(
        self,
        session: SessionState,
        shot_id: str,
        shot_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Generate video for a single shot without running full Agent 10.

        Args:
            session: Session state (must have Phase 2 completed)
            shot_id: Shot identifier (e.g., "SHOT_1_1")
            shot_type: "parent" or "child"
            progress_callback: Optional callback for progress updates

        Returns:
            Video generation result for the shot

        Raises:
            ValueError: If Phase 2 not complete or shot not found
        """
        logger.info(f"Generating video for single shot: {shot_id} ({shot_type})")

        # Verify Phase 2 is complete
        if not self._phase_2_complete(session):
            raise ValueError("Phase 2 (image generation) must be completed before generating videos")

        # Get session directory
        session_dir = Path(self.session_manager.base_directory) / session.session_id

        # Prepare input data from session
        input_data = {
            "parent_shots": session.agents.get("agent_7", {}).output_data.get("parent_shots", []) if "agent_7" in session.agents else [],
            "child_shots": session.agents.get("agent_9", {}).output_data.get("child_shots", []) if "agent_9" in session.agents else [],
            "shot_breakdown": session.agents.get("agent_3", {}).output_data if "agent_3" in session.agents else {},
            "character_data": session.agents.get("agent_5", {}).output_data if "agent_5" in session.agents else {}
        }

        # Progress update
        if progress_callback:
            progress_callback(f"Generating video for {shot_id}...", 0.1)

        # Import and instantiate Agent 10
        from agents.agent_10_video_dialogue import VideoDialogueAgent

        agent_config = self.config.get("agents", {}).get("agent_10", {})
        agent = VideoDialogueAgent(
            gemini_client=self.gemini_client,
            config=agent_config,
            session_dir=session_dir
        )

        # Generate video
        try:
            video_data = agent.generate_single_shot_video(
                shot_id=shot_id,
                shot_type=shot_type,
                input_data=input_data
            )

            if progress_callback:
                if video_data.get("status") == "failed":
                    progress_callback(f"Video generation failed: {video_data.get('error')}", 1.0, error=True)
                else:
                    progress_callback(f"Video generated successfully for {shot_id}", 1.0)

            logger.info(f"Successfully generated video for {shot_id}")
            return video_data

        except Exception as e:
            logger.error(f"Failed to generate video for {shot_id}: {str(e)}")
            if progress_callback:
                progress_callback(f"Failed to generate video: {str(e)}", 1.0, error=True)
            raise

    def _get_agent_input(self, session: SessionState, agent_name: str) -> Any:
        """
        Get input data for an agent.
        Phase 2 agents require data from multiple previous agents.

        Args:
            session: Current session state
            agent_name: Agent name

        Returns:
            Input data for the agent
        """
        if agent_name == session.start_agent:
            # First agent uses original input
            return session.input_data

        # Phase 2 agents have special input requirements
        if agent_name == "agent_5":
            # Character Creator needs Agent 2 (scene breakdown) and Agent 3 (shot breakdown)
            return {
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3")
            }

        elif agent_name == "agent_6":
            # Parent Image Generator needs Agent 2, 3, 4, and 5
            return {
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "character_grids": self._get_agent_output(session, "agent_5").get("character_grids", [])
            }

        elif agent_name == "agent_7":
            # Parent Verification (with regeneration capability) needs same data as Agent 6
            return {
                "parent_shots": self._get_agent_output(session, "agent_6").get("parent_shots", []),
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "character_grids": self._get_agent_output(session, "agent_5").get("character_grids", [])
            }

        elif agent_name == "agent_8":
            # Child Image Generator needs Agent 2, 3, 4, 5, and verified Agent 7
            return {
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "parent_shots": self._get_agent_output(session, "agent_7").get("parent_shots", []),
                "character_grids": self._get_agent_output(session, "agent_5").get("character_grids", [])
            }

        elif agent_name == "agent_9":
            # Child Verification (with regeneration capability) needs same data as Agent 8
            return {
                "child_shots": self._get_agent_output(session, "agent_8").get("child_shots", []),
                "parent_shots": self._get_agent_output(session, "agent_7").get("parent_shots", []),
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "character_grids": self._get_agent_output(session, "agent_5").get("character_grids", [])
            }

        elif agent_name == "agent_10":
            # Video Dialogue Generator needs both parent and child shots plus context
            return {
                "parent_shots": self._get_agent_output(session, "agent_7").get("parent_shots", []),
                "child_shots": self._get_agent_output(session, "agent_9").get("child_shots", []),
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "character_data": self._get_agent_output(session, "agent_5")  # Full agent_5 output with characters list
            }

        elif agent_name == "agent_11":
            # Intelligent Video Editor needs videos from Agent 10 plus scene/shot context
            return {
                "videos": self._get_agent_output(session, "agent_10").get("videos", []),
                "scene_breakdown": self._get_agent_output(session, "agent_2"),
                "shot_breakdown": self._get_agent_output(session, "agent_3"),
                "shot_grouping": self._get_agent_output(session, "agent_4"),
                "session_metadata": {
                    "session_id": session.session_id,
                    "session_name": session.session_name
                }
            }

        # Phase 1 agents: Get output from previous agent
        agent_idx = self.agent_order.index(agent_name)
        prev_agent = self.agent_order[agent_idx - 1]

        if prev_agent not in session.agents:
            raise ValueError(
                f"Cannot run {agent_name}: previous agent {prev_agent} has not been executed"
            )

        prev_output = session.agents[prev_agent].output_data

        if prev_output is None:
            raise ValueError(
                f"Cannot run {agent_name}: previous agent {prev_agent} produced no output"
            )

        return prev_output

    def _get_agent_output(self, session: SessionState, agent_name: str) -> Any:
        """Helper to get agent output with error checking."""
        if agent_name not in session.agents:
            raise ValueError(f"Agent {agent_name} has not been executed")

        output = session.agents[agent_name].output_data

        if output is None:
            raise ValueError(f"Agent {agent_name} produced no output")

        return output

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session by ID."""
        return self.session_manager.load_session(session_id)

    def list_sessions(self, limit: int = 20):
        """List recent sessions."""
        return self.session_manager.list_sessions(limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.session_manager.delete_session(session_id)
