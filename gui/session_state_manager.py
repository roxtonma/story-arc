"""
Session State Manager
Centralized management of Streamlit session state for Story Architect GUI.
"""

from typing import Optional
import streamlit as st

from core.pipeline import Pipeline
from core.validators import SessionState


class SessionStateManager:
    """Manages Streamlit session state for the Story Architect application."""

    @staticmethod
    def init():
        """Initialize all session state variables."""
        # Core pipeline instance
        if "pipeline" not in st.session_state:
            try:
                st.session_state.pipeline = Pipeline()
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {str(e)}")
                st.stop()

        # Active session tracking
        if "active_session_id" not in st.session_state:
            st.session_state.active_session_id = None

        if "active_session" not in st.session_state:
            st.session_state.active_session = None

        # Execution state
        if "running" not in st.session_state:
            st.session_state.running = False

        if "current_phase" not in st.session_state:
            st.session_state.current_phase = None

        # UI state
        if "selected_tab" not in st.session_state:
            st.session_state.selected_tab = "home"

        # Retry state
        if "retry_agent" not in st.session_state:
            st.session_state.retry_agent = None

        if "retry_input" not in st.session_state:
            st.session_state.retry_input = None

        # Shot-level video generation state
        if "generating_video_for_shot" not in st.session_state:
            st.session_state.generating_video_for_shot = None

    @staticmethod
    def get_pipeline() -> Pipeline:
        """Get the Pipeline instance."""
        return st.session_state.pipeline

    @staticmethod
    def get_active_session() -> Optional[SessionState]:
        """Get the currently active session."""
        return st.session_state.active_session

    @staticmethod
    def set_active_session(session: SessionState):
        """Set the currently active session."""
        st.session_state.active_session = session
        st.session_state.active_session_id = session.session_id if session else None

    @staticmethod
    def load_session(session_id: str) -> Optional[SessionState]:
        """Load a session by ID and set it as active."""
        try:
            pipeline = SessionStateManager.get_pipeline()
            session = pipeline.get_session(session_id)
            if session:
                SessionStateManager.set_active_session(session)
                return session
            return None
        except Exception as e:
            st.error(f"Failed to load session: {str(e)}")
            return None

    @staticmethod
    def create_new_session(input_data: str, start_agent: str, session_name: Optional[str] = None) -> Optional[SessionState]:
        """Create a new session and set it as active."""
        try:
            pipeline = SessionStateManager.get_pipeline()
            session = pipeline.create_session(
                input_data=input_data,
                start_agent=start_agent,
                session_name=session_name
            )
            SessionStateManager.set_active_session(session)
            return session
        except Exception as e:
            st.error(f"Failed to create session: {str(e)}")
            return None

    @staticmethod
    def is_running() -> bool:
        """Check if pipeline is currently running."""
        return st.session_state.running

    @staticmethod
    def set_running(running: bool):
        """Set pipeline running state."""
        st.session_state.running = running

    @staticmethod
    def get_current_phase() -> Optional[str]:
        """Get the currently executing phase."""
        return st.session_state.current_phase

    @staticmethod
    def set_current_phase(phase: Optional[str]):
        """Set the currently executing phase."""
        st.session_state.current_phase = phase

    @staticmethod
    def refresh_active_session():
        """Reload the active session from disk."""
        if st.session_state.active_session_id:
            SessionStateManager.load_session(st.session_state.active_session_id)

    @staticmethod
    def clear_active_session():
        """Clear the active session."""
        st.session_state.active_session = None
        st.session_state.active_session_id = None

    @staticmethod
    def get_selected_tab() -> str:
        """Get the currently selected tab."""
        return st.session_state.selected_tab

    @staticmethod
    def set_selected_tab(tab: str):
        """Set the currently selected tab."""
        st.session_state.selected_tab = tab

    @staticmethod
    def set_retry_agent(agent_name: Optional[str], input_data: Optional[str] = None):
        """Set the agent to retry with optional input data."""
        st.session_state.retry_agent = agent_name
        st.session_state.retry_input = input_data

    @staticmethod
    def get_retry_agent() -> Optional[tuple]:
        """Get the agent to retry and its input data."""
        return (st.session_state.retry_agent, st.session_state.retry_input)

    @staticmethod
    def clear_retry_agent():
        """Clear retry state."""
        st.session_state.retry_agent = None
        st.session_state.retry_input = None

    @staticmethod
    def set_generating_video_for_shot(shot_id: Optional[str]):
        """Set the shot currently generating video."""
        st.session_state.generating_video_for_shot = shot_id

    @staticmethod
    def get_generating_video_for_shot() -> Optional[str]:
        """Get the shot currently generating video."""
        return st.session_state.generating_video_for_shot

    @staticmethod
    def clear_generating_video_for_shot():
        """Clear video generation state."""
        st.session_state.generating_video_for_shot = None
