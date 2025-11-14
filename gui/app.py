"""
Story Architect - Streamlit GUI
Main application interface for the multi-agent pipeline.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from core.pipeline import Pipeline
from core.validators import SessionState
from core.export import generate_notion_markdown

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")


# Page configuration
st.set_page_config(
    page_title="Story Architect",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state."""
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = Pipeline()
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            st.stop()

    if "current_session" not in st.session_state:
        st.session_state.current_session = None

    if "running" not in st.session_state:
        st.session_state.running = False


def main():
    """Main application."""
    init_session_state()

    st.title("🎬 Story Architect")
    st.markdown("*AI-Powered Multi-Agent System for Script-to-Shot Conversion*")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")

        page = st.radio(
            "Select Page",
            ["New Project", "Resume Session", "Session History"],
            key="page_selector"
        )

        st.divider()

        # API Key Status
        st.subheader("API Status")
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("✓ API Key Configured")
        else:
            st.error("✗ API Key Not Found")
            st.info("Set GOOGLE_API_KEY in .env file")

    # Main content
    if page == "New Project":
        render_new_project_page()
    elif page == "Resume Session":
        render_resume_session_page()
    elif page == "Session History":
        render_session_history_page()


def render_new_project_page():
    """Render the new project page."""
    st.header("Create New Project")

    # Starting point selection
    col1, col2 = st.columns(2)

    with col1:
        start_option = st.radio(
            "Choose Starting Point",
            ["Start from scratch (Agent 1)", "I have a screenplay (Agent 2)"],
            help="Agent 1 generates screenplay from logline/story. Agent 2 starts with existing screenplay."
        )

    with col2:
        session_name = st.text_input(
            "Project Name (Optional)",
            placeholder="e.g., My Action Movie",
            help="Give your project a memorable name"
        )

    # Input area
    st.subheader("Input")

    if "Start from scratch" in start_option:
        input_label = "Enter your logline, story concept, or rough script:"
        input_help = "Provide a story idea that Agent 1 will convert into a screenplay"
        start_agent = "agent_1"
    else:
        input_label = "Paste your screenplay:"
        input_help = "Provide a formatted screenplay (with INT./EXT. scene headings)"
        start_agent = "agent_2"

    input_data = st.text_area(
        input_label,
        height=300,
        help=input_help,
        placeholder="Enter your content here..."
    )

    # Action buttons
    col1, col2 = st.columns([1, 4])

    with col1:
        run_button = st.button(
            "🚀 Run Pipeline",
            type="primary",
            disabled=not input_data or st.session_state.running,
            use_container_width=True
        )

    if run_button and input_data:
        # Create new session
        try:
            session = st.session_state.pipeline.create_session(
                input_data=input_data.strip(),
                start_agent=start_agent,
                session_name=session_name if session_name else None
            )

            st.session_state.current_session = session
            st.session_state.running = True

            # Run pipeline
            run_pipeline_with_progress(session)

        except Exception as e:
            st.error(f"Failed to create session: {str(e)}")
            st.session_state.running = False


def run_pipeline_with_progress(session: SessionState):
    """Run pipeline with progress indicators."""
    st.subheader("Pipeline Execution")

    # Progress container
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Define progress callback
        def progress_callback(message, progress, error=False):
            progress_bar.progress(min(progress, 1.0))
            if error:
                status_text.error(message)
            else:
                status_text.info(message)

        try:
            # Run pipeline
            updated_session = st.session_state.pipeline.run_pipeline(
                session=session,
                progress_callback=progress_callback
            )

            st.session_state.current_session = updated_session
            st.session_state.running = False

            st.success("✅ Pipeline completed successfully!")

            # Display results
            display_session_results(updated_session)

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.session_state.running = False

            # Show partial results if available
            if session.agents:
                st.warning("Showing partial results from completed agents:")
                display_session_results(session)


def display_session_results(session: SessionState):
    """Display session results in tabs."""
    st.subheader("Results")

    # Create tabs for each agent
    tabs = st.tabs(["Agent 1: Screenplay", "Agent 2: Scenes", "Agent 3: Shots", "Agent 4: Grouping"])

    # Agent 1 Output
    with tabs[0]:
        if "agent_1" in session.agents:
            agent_output = session.agents["agent_1"]
            if agent_output.status == "completed":
                st.markdown("### Generated Screenplay")
                st.text_area(
                    "Screenplay Output",
                    value=agent_output.output_data,
                    height=400,
                    key="agent_1_output"
                )

                # Download button
                st.download_button(
                    "📥 Download Screenplay",
                    data=agent_output.output_data,
                    file_name=f"{session.session_id}_screenplay.txt",
                    mime="text/plain"
                )
            else:
                st.warning(f"Agent 1 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)
        else:
            st.info("Agent 1 not executed (started from Agent 2)")

    # Agent 2 Output
    with tabs[1]:
        if "agent_2" in session.agents:
            agent_output = session.agents["agent_2"]
            if agent_output.status == "completed":
                st.markdown("### Scene Breakdown")

                # Display as formatted JSON
                st.json(agent_output.output_data)

                # Download button
                st.download_button(
                    "📥 Download Scene Breakdown",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_scenes.json",
                    mime="application/json"
                )

                # Scene summary
                total_scenes = agent_output.output_data.get("total_scenes", 0)
                st.info(f"Total Scenes: {total_scenes}")
            else:
                st.warning(f"Agent 2 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)
        else:
            st.info("Agent 2 not yet executed")

    # Agent 3 Output
    with tabs[2]:
        if "agent_3" in session.agents:
            agent_output = session.agents["agent_3"]
            if agent_output.status == "completed":
                st.markdown("### Shot Breakdown")

                st.json(agent_output.output_data)

                # Download button
                st.download_button(
                    "📥 Download Shot Breakdown",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_shots.json",
                    mime="application/json"
                )

                # Shot summary
                total_shots = agent_output.output_data.get("total_shots", 0)
                st.info(f"Total Shots: {total_shots}")
            else:
                st.warning(f"Agent 3 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)
        else:
            st.info("Agent 3 not yet executed")

    # Agent 4 Output
    with tabs[3]:
        if "agent_4" in session.agents:
            agent_output = session.agents["agent_4"]
            if agent_output.status == "completed":
                st.markdown("### Shot Grouping")

                st.json(agent_output.output_data)

                # Download buttons
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        "📥 Download as JSON",
                        data=json.dumps(agent_output.output_data, indent=2),
                        file_name=f"{session.session_id}_grouped_shots.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    # Generate Notion-compatible Markdown
                    # Need Agent 3 data for shot lookup (Agent 4 only has references)
                    agent3_output = session.agents.get("agent_3", {})
                    if agent3_output and agent3_output.output_data:
                        notion_markdown = generate_notion_markdown(
                            agent_output.output_data,
                            agent3_output.output_data
                        )
                        st.download_button(
                            "📥 Download as Notion Markdown",
                            data=notion_markdown,
                            file_name=f"{session.session_id}_grouped_shots.md",
                            mime="text/markdown",
                            help="Notion-compatible Markdown with collapsible shot hierarchy",
                            use_container_width=True
                        )
                    else:
                        st.warning("Notion export requires Agent 3 output")

                # Grouping summary
                total_parent = agent_output.output_data.get("total_parent_shots", 0)
                total_child = agent_output.output_data.get("total_child_shots", 0)
                st.info(f"Parent Shots: {total_parent} | Child Shots: {total_child}")
            else:
                st.warning(f"Agent 4 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)
        else:
            st.info("Agent 4 not yet executed")


def render_resume_session_page():
    """Render the resume session page."""
    st.header("Resume Session")

    # Load recent sessions
    sessions = st.session_state.pipeline.list_sessions(limit=10)

    if not sessions:
        st.info("No sessions found. Create a new project to get started.")
        return

    # Session selector
    session_options = {
        f"{s['session_name']} ({s['session_id']})": s['session_id']
        for s in sessions
    }

    selected = st.selectbox(
        "Select a session to resume",
        options=list(session_options.keys())
    )

    if selected:
        session_id = session_options[selected]

        # Load session
        session = st.session_state.pipeline.get_session(session_id)

        if session:
            # Display session info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", session.status)
            with col2:
                st.metric("Current Agent", session.current_agent)
            with col3:
                st.metric("Start Agent", session.start_agent)

            st.divider()

            # Display current results
            display_session_results(session)

            st.divider()

            # Resume options
            st.subheader("Resume Options")

            resume_from = st.selectbox(
                "Resume from agent",
                options=["agent_1", "agent_2", "agent_3", "agent_4"]
            )

            if st.button("Resume Pipeline", type="primary"):
                try:
                    st.session_state.current_session = session
                    st.session_state.running = True

                    # Resume pipeline
                    updated_session = st.session_state.pipeline.resume_from_agent(
                        session=session,
                        agent_name=resume_from
                    )

                    st.session_state.current_session = updated_session
                    st.session_state.running = False

                    st.success("Pipeline resumed and completed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to resume pipeline: {str(e)}")
                    st.session_state.running = False


def render_session_history_page():
    """Render the session history page."""
    st.header("Session History")

    # Load all sessions
    sessions = st.session_state.pipeline.list_sessions(limit=50)

    if not sessions:
        st.info("No sessions found.")
        return

    # Display sessions in a table
    for session in sessions:
        with st.expander(f"📁 {session['session_name']} - {session['status'].upper()}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Session ID:** {session['session_id']}")
                st.write(f"**Status:** {session['status']}")
                st.write(f"**Current Agent:** {session['current_agent']}")

            with col2:
                st.write(f"**Created:** {session['created_at']}")
                st.write(f"**Updated:** {session['updated_at']}")

            # Actions
            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"View Session", key=f"view_{session['session_id']}"):
                    loaded_session = st.session_state.pipeline.get_session(session['session_id'])
                    if loaded_session:
                        st.session_state.current_session = loaded_session
                        display_session_results(loaded_session)

            with col2:
                if st.button(f"Delete Session", key=f"delete_{session['session_id']}", type="secondary"):
                    if st.session_state.pipeline.delete_session(session['session_id']):
                        st.success("Session deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete session")


if __name__ == "__main__":
    main()
