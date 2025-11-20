"""
Session Manager
Handles session persistence, state management, and resume capability.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from core.validators import SessionState, AgentOutput


class SessionManager:
    """Manages session state, persistence, and resumability."""

    def __init__(self, base_directory: str = "outputs/projects"):
        """
        Initialize session manager.

        Args:
            base_directory: Base directory for storing sessions
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized SessionManager with base directory: {self.base_directory}")

    def create_session(
        self,
        input_data: str,
        start_agent: str,
        session_name: Optional[str] = None
    ) -> SessionState:
        """
        Create a new session.

        Args:
            input_data: Initial input data
            start_agent: Starting agent ('agent_1' or 'agent_2')
            session_name: Optional user-friendly name

        Returns:
            New SessionState object
        """
        timestamp = datetime.now()
        session_id = timestamp.strftime("%Y%m%d_%H%M%S")

        if not session_name:
            session_name = f"Session {session_id}"

        session = SessionState(
            session_id=session_id,
            session_name=session_name,
            created_at=timestamp.isoformat(),
            updated_at=timestamp.isoformat(),
            start_agent=start_agent,
            current_agent=start_agent,
            agents={},
            input_data=input_data,
            status="in_progress"
        )

        # Create session directory
        session_dir = self._get_session_directory(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save initial session state
        self.save_session(session)

        logger.info(f"Created new session: {session_id} ({session_name})")
        return session

    def save_session(self, session: SessionState) -> None:
        """
        Save session state to disk.

        Args:
            session: SessionState to save
        """
        session_dir = self._get_session_directory(session.session_id)
        session_file = session_dir / "session_state.json"

        # Update timestamp
        session.updated_at = datetime.now().isoformat()

        # Save as JSON
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session.model_dump(), f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved session state: {session.session_id}")

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load session state from disk.

        Args:
            session_id: Session ID to load

        Returns:
            SessionState object or None if not found
        """
        session_dir = self._get_session_directory(session_id)
        session_file = session_dir / "session_state.json"

        if not session_file.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session = SessionState(**data)
            logger.info(f"Loaded session: {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None

    def update_agent_output(
        self,
        session: SessionState,
        agent_name: str,
        output_data: Any,
        status: str = "completed",
        error_message: Optional[str] = None,
        retry_count: int = 0
    ) -> SessionState:
        """
        Update agent output in session.

        Args:
            session: SessionState to update
            agent_name: Name of the agent
            output_data: Output data from agent
            status: Agent status
            error_message: Error message if failed
            retry_count: Number of retry attempts

        Returns:
            Updated SessionState
        """
        agent_output = AgentOutput(
            agent_name=agent_name,
            status=status,
            output_data=output_data,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            retry_count=retry_count
        )

        session.agents[agent_name] = agent_output
        session.current_agent = agent_name

        # Update overall session status
        if status == "failed":
            session.status = "failed"
        elif status == "completed":
            # Check if this is the last agent (Phase 3 completion)
            if agent_name == "agent_10":
                session.status = "completed"

        # Save agent output to separate file
        self._save_agent_output(session.session_id, agent_name, output_data)

        # Save updated session state
        self.save_session(session)

        logger.info(f"Updated agent output: {agent_name} -> {status}")
        return session

    def get_agent_output(
        self,
        session: SessionState,
        agent_name: str
    ) -> Optional[Any]:
        """
        Get output from a specific agent.

        Args:
            session: SessionState
            agent_name: Name of the agent

        Returns:
            Agent output data or None
        """
        if agent_name in session.agents:
            return session.agents[agent_name].output_data
        return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List all available sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        sessions = []

        for session_dir in sorted(self.base_directory.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session_state.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sessions.append({
                    "session_id": data["session_id"],
                    "session_name": data.get("session_name", "Unnamed"),
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "status": data["status"],
                    "current_agent": data["current_agent"]
                })

                if len(sessions) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to read session in {session_dir}: {str(e)}")
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its files.

        Args:
            session_id: Session ID to delete

        Returns:
            True if successful, False otherwise
        """
        session_dir = self._get_session_directory(session_id)

        if not session_dir.exists():
            logger.warning(f"Session not found: {session_id}")
            return False

        try:
            # Delete all files in session directory
            for file in session_dir.rglob("*"):
                if file.is_file():
                    file.unlink()

            # Delete directory
            session_dir.rmdir()
            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False

    def _get_session_directory(self, session_id: str) -> Path:
        """Get session directory path."""
        return self.base_directory / session_id

    def _save_agent_output(
        self,
        session_id: str,
        agent_name: str,
        output_data: Any
    ) -> None:
        """
        Save agent output to separate file.

        Args:
            session_id: Session ID
            agent_name: Agent name
            output_data: Output data to save
        """
        session_dir = self._get_session_directory(session_id)

        # Determine file extension based on data type
        if isinstance(output_data, (dict, list)):
            filename = f"{agent_name}_output.json"
            with open(session_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            filename = f"{agent_name}_output.txt"
            with open(session_dir / filename, 'w', encoding='utf-8') as f:
                f.write(str(output_data))

        logger.debug(f"Saved {agent_name} output to {filename}")

    def can_resume_from_agent(
        self,
        session: SessionState,
        agent_name: str
    ) -> bool:
        """
        Check if session can resume from a specific agent.

        Args:
            session: SessionState
            agent_name: Agent to resume from

        Returns:
            True if can resume, False otherwise
        """
        # Define agent order (Phase 1 + Phase 2 + Phase 3)
        agent_order = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6", "agent_7", "agent_8", "agent_9", "agent_10"]

        # Can't resume from an agent before the start agent
        if agent_order.index(agent_name) < agent_order.index(session.start_agent):
            return False

        # Can resume from start agent
        if agent_name == session.start_agent:
            return True

        # Check if previous agent completed
        prev_agent_idx = agent_order.index(agent_name) - 1
        prev_agent = agent_order[prev_agent_idx]

        if prev_agent in session.agents:
            return session.agents[prev_agent].status == "completed"

        return False
