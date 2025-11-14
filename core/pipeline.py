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
        self.agent_order = ["agent_1", "agent_2", "agent_3", "agent_4"]

        logger.info("Pipeline initialized successfully")

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

        logger.info(f"Initialized {len(agents)} agents")
        return agents

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
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")

        agent = self.agents[agent_name]
        max_retries = self.config["app"]["max_retries"]

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

    def _get_agent_input(self, session: SessionState, agent_name: str) -> Any:
        """
        Get input data for an agent.

        Args:
            session: Current session state
            agent_name: Agent name

        Returns:
            Input data for the agent
        """
        if agent_name == session.start_agent:
            # First agent uses original input
            return session.input_data

        # Get output from previous agent
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

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session by ID."""
        return self.session_manager.load_session(session_id)

    def list_sessions(self, limit: int = 20):
        """List recent sessions."""
        return self.session_manager.list_sessions(limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.session_manager.delete_session(session_id)
