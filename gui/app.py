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
from gui.session_state_manager import SessionStateManager

# Load environment variables (fallback for local development)
load_dotenv()

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")


def get_secret(key: str, default=None):
    """
    Get secret from Streamlit secrets (priority) or environment variables (fallback).

    Args:
        key: The secret key to retrieve
        default: Default value if not found

    Returns:
        Secret value or default
    """
    # Try Streamlit secrets first (for Cloud deployment)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, KeyError):
        pass

    # Fallback to environment variables (for local development)
    return os.getenv(key, default)


# Page configuration
st.set_page_config(
    page_title="Story Architect",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for production tool aesthetic
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0A0A0F;
    }

    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #13131A;
        padding: 8px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E2E;
        border-radius: 6px;
        padding: 8px 16px;
        color: #A0A0A8;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #A855F7;
        color: #FFFFFF;
        border-color: #A855F7;
    }

    /* Enhance expanders */
    .streamlit-expanderHeader {
        background-color: #1E1E2E;
        border-radius: 6px;
        border: 1px solid #2A2A38;
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        border-color: #A855F7;
        background-color: #252530;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #2A2A38;
        transition: all 0.2s ease;
        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #A855F7;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.2);
    }

    /* Primary button enhancement */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #A855F7 0%, #8B5CF6 100%);
        border: none;
    }

    /* Success/error/warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 6px;
        border-left-width: 4px;
    }

    /* Image and Video containers */
    .stImage, .stVideo {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #A855F7 0%, #8B5CF6 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0D0D12;
        border-right: 1px solid #2A2A38;
    }

    /* Text area improvements */
    .stTextArea textarea {
        background-color: #1E1E2E;
        border: 1px solid #2A2A38;
        border-radius: 6px;
        color: #F5F5F7;
    }

    .stTextArea textarea:focus {
        border-color: #A855F7;
        box-shadow: 0 0 0 1px #A855F7;
    }

    /* JSON viewer */
    .stJson {
        background-color: #1E1E2E;
        border-radius: 6px;
        border: 1px solid #2A2A38;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state using SessionStateManager."""
    SessionStateManager.init()


def main():
    """Main application."""
    init_session_state()

    st.title("🎬 Story Architect")
    st.markdown("*AI-Powered Multi-Agent System for Story-to-Video Generation*")

    # Sidebar
    with st.sidebar:
        st.header("Quick Info")

        # API Key Status
        st.subheader("API Status")

        # Check for Gemini API Key
        gemini_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if gemini_key:
            st.success("✓ Gemini API")
        else:
            st.error("✗ Gemini API")

        # Check for FAL API Key
        fal_key = get_secret("FAL_KEY")
        if fal_key:
            st.success("✓ FAL.ai API")
        else:
            st.warning("✗ FAL.ai API")

        # Show configuration source
        try:
            if "GEMINI_API_KEY" in st.secrets or "GOOGLE_API_KEY" in st.secrets:
                st.caption("🔐 Streamlit Secrets")
            else:
                st.caption("📁 .env file")
        except FileNotFoundError:
            st.caption("📁 .env file")

        st.divider()

        # Active Session Info
        active_session = SessionStateManager.get_active_session()
        if active_session:
            st.subheader("Active Session")
            st.caption(f"**{active_session.session_name}**")
            st.caption(f"ID: {active_session.session_id[:8]}...")
            st.caption(f"Status: {active_session.status}")

            # Phase status indicators
            pipeline = SessionStateManager.get_pipeline()

            phase1_done = all(
                f"agent_{i}" in active_session.agents and
                active_session.agents[f"agent_{i}"].status == "completed"
                for i in range(1, 5)
            )
            phase2_done = all(
                f"agent_{i}" in active_session.agents and
                active_session.agents[f"agent_{i}"].status == "completed"
                for i in range(5, 10)
            )
            phase3_done = "agent_10" in active_session.agents and \
                         active_session.agents["agent_10"].status == "completed"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("📝" if phase1_done else "⏳")
            with col2:
                st.caption("🎨" if phase2_done else "⏳")
            with col3:
                st.caption("🎬" if phase3_done else "⏳")

    # Main content - Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "✨ New Project",
        "⏯️ Resume Session",
        "📚 History",
        "⚙️ Settings"
    ])

    with tab1:
        render_home_page()

    with tab2:
        render_new_project_page()

    with tab3:
        render_resume_session_page()

    with tab4:
        render_session_history_page()

    with tab5:
        render_configuration_page()


def render_home_page():
    """Render the home/welcome page."""
    st.header("Welcome to Story Architect")

    st.markdown("""
    ### Transform Stories into Cinematic Visuals

    Story Architect is an AI-powered system that converts story concepts into complete visual productions:

    **📝 Phase 1: Script-to-Shot**
    - Generate screenplay from story concept
    - Break down into detailed scenes
    - Create shot-by-shot breakdown
    - Optimize shot hierarchy

    **🎨 Phase 2: Image Generation**
    - Create consistent characters
    - Generate parent shot images
    - Create child shot variations
    - AI-powered verification

    **🎬 Phase 3: Video Generation**
    - Transform images into videos
    - Production brief generation
    - 4-8 second video clips

    ---

    ### Quick Start

    1. **New Project** - Start with a story concept or screenplay
    2. **Configure** - Ensure your API keys are set (check sidebar)
    3. **Generate** - Let the pipeline create your visual story
    4. **Export** - Download in HTML, Notion, or complete archive

    ---

    ### System Requirements

    **Required APIs:**
    - ✓ Google Gemini API (text generation)
    - ✓ FAL.ai API (image/video generation)

    **Optional:**
    - Google Cloud Vertex AI (enterprise deployment)

    Ready to begin? Click **New Project** in the sidebar!
    """)

    # Quick stats if there are sessions
    pipeline = SessionStateManager.get_pipeline()
    sessions = pipeline.list_sessions(limit=5)
    if sessions:
        st.divider()
        st.subheader("Recent Activity")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", len(sessions))
        with col2:
            completed = sum(1 for s in sessions if s['status'] == 'completed')
            st.metric("Completed", completed)
        with col3:
            in_progress = sum(1 for s in sessions if s['status'] == 'running')
            st.metric("In Progress", in_progress)


def render_configuration_page():
    """Render the configuration management page."""
    st.header("⚙️ Configuration")

    st.markdown("""
    Manage your API keys and system configuration. Keys can be set in:
    - `.streamlit/secrets.toml` (local development)
    - Streamlit Cloud Secrets (deployed apps)
    - `.env` file (fallback)
    """)

    st.divider()

    # API Keys Section
    st.subheader("API Keys Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Google Gemini")
        gemini_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if gemini_key:
            st.success("✓ Configured")
            masked_key = f"{gemini_key[:8]}...{gemini_key[-4:]}" if len(gemini_key) > 12 else "***"
            st.code(masked_key)
        else:
            st.error("✗ Not Found")
            st.markdown("[Get API Key →](https://aistudio.google.com/apikey)")

    with col2:
        st.markdown("#### FAL.ai")
        fal_key = get_secret("FAL_KEY")
        if fal_key:
            st.success("✓ Configured")
            masked_key = f"{fal_key[:8]}...{fal_key[-4:]}" if len(fal_key) > 12 else "***"
            st.code(masked_key)
        else:
            st.error("✗ Not Found")
            st.markdown("[Get API Key →](https://fal.ai/dashboard/keys)")

    st.divider()

    # Google Cloud / Vertex AI
    st.subheader("Google Cloud / Vertex AI (Optional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        project = get_secret("GOOGLE_CLOUD_PROJECT")
        if project:
            st.success("✓ Project ID")
            st.code(project)
        else:
            st.info("Not configured")

    with col2:
        location = get_secret("GOOGLE_CLOUD_LOCATION")
        if location:
            st.success("✓ Location")
            st.code(location)
        else:
            st.info("Not configured")

    with col3:
        credentials = get_secret("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials:
            st.success("✓ Credentials File")
            st.code(credentials)
        else:
            st.info("Not configured")

    st.divider()

    # Configuration Guide
    with st.expander("📖 How to Configure Secrets"):
        st.markdown("""
        ### Local Development

        Create `.streamlit/secrets.toml` in your project root:

        ```toml
        GEMINI_API_KEY = "your-key-here"
        FAL_KEY = "your-key-here"
        ```

        ### Streamlit Cloud Deployment

        1. Deploy your app to Streamlit Cloud
        2. Go to app Settings → Secrets
        3. Paste your secrets in TOML format
        4. Click Save

        ### Environment Variables (.env)

        Alternatively, create a `.env` file:

        ```bash
        GEMINI_API_KEY=your-key-here
        FAL_KEY=your-key-here
        ```

        **Note:** Never commit API keys to Git!
        """)


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
        placeholder="Enter your content here...",
        key="new_project_input"
    )

    # Execution mode selection
    st.subheader("Execution Mode")
    exec_mode = st.radio(
        "Choose which phases to execute:",
        [
            "🎬 Full Pipeline (All 3 Phases)",
            "📝 Phase 1: Script to Shot (Agents 1-4)",
            "🎨 Phase 2: Image Generation (Agents 5-9)",
            "🎥 Phase 3: Video Generation (Agent 10)"
        ],
        help="Select which phase(s) to run. Phase 2 requires Phase 1, Phase 3 requires Phase 2."
    )

    # Action buttons
    col1, col2 = st.columns([1, 4])

    with col1:
        run_button = st.button(
            "🚀 Run Pipeline",
            type="primary",
            disabled=not input_data or SessionStateManager.is_running(),
            use_container_width=True
        )

    if run_button and input_data:
        # Create new session
        try:
            session = SessionStateManager.create_new_session(
                input_data=input_data.strip(),
                start_agent=start_agent,
                session_name=session_name if session_name else None
            )

            if not session:
                st.error("Failed to create session")
                return

            SessionStateManager.set_running(True)

            # Determine execution mode
            if "Full Pipeline" in exec_mode:
                phase = "full"
            elif "Phase 1" in exec_mode:
                phase = "phase_1"
            elif "Phase 2" in exec_mode:
                phase = "phase_2"
            elif "Phase 3" in exec_mode:
                phase = "phase_3"
            else:
                phase = "full"

            # Run pipeline
            run_pipeline_with_progress(session, phase=phase)

        except Exception as e:
            st.error(f"Failed to create session: {str(e)}")
            SessionStateManager.set_running(False)


def run_pipeline_with_progress(session: SessionState, phase: str = "full"):
    """
    Run pipeline with progress indicators.

    Args:
        session: Session state
        phase: Phase to execute ("full", "phase_1", "phase_2", "phase_3")
    """
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
            pipeline = SessionStateManager.get_pipeline()

            # Run selected phase
            if phase == "phase_1":
                updated_session = pipeline.run_phase_1(session, progress_callback)
                success_msg = "✅ Phase 1 completed successfully!"
            elif phase == "phase_2":
                updated_session = pipeline.run_phase_2(session, progress_callback)
                success_msg = "✅ Phase 2 completed successfully!"
            elif phase == "phase_3":
                updated_session = pipeline.run_phase_3(session, progress_callback)
                success_msg = "✅ Phase 3 completed successfully!"
            else:
                # Full pipeline
                updated_session = pipeline.run_pipeline(session, progress_callback)
                success_msg = "✅ Full pipeline completed successfully!"

            SessionStateManager.set_active_session(updated_session)
            SessionStateManager.set_running(False)

            st.success(success_msg)

            # Display results
            display_session_results(updated_session)

        except Exception as e:
            st.error(f"Execution failed: {str(e)}")
            SessionStateManager.set_running(False)

            # Refresh session from disk to get partial results
            SessionStateManager.refresh_active_session()
            session = SessionStateManager.get_active_session()

            # Show partial results if available
            if session and session.agents:
                st.warning("Showing partial results from completed agents:")
                display_session_results(session)


def render_retry_ui(session: SessionState, agent_name: str, agent_output):
    """Render retry UI for failed agents."""
    if agent_output.status in ["failed", "soft_failure"]:
        with st.expander("🔄 Retry Agent"):
            st.warning(f"**Status:** {agent_output.status}")
            if agent_output.error_message:
                st.error(f"**Error:** {agent_output.error_message}")

            st.markdown("---")
            st.markdown("**Edit input before retry (optional):**")

            # Get original input
            pipeline = SessionStateManager.get_pipeline()
            try:
                original_input = pipeline._get_agent_input(session, agent_name)
                input_json = json.dumps(original_input, indent=2)
            except:
                input_json = "{}"

            # Editable input
            edited_input = st.text_area(
                "Agent Input (JSON):",
                value=input_json,
                height=200,
                key=f"retry_input_{agent_name}",
                help="Modify the input data if needed, or leave as-is to retry with original input"
            )

            col1, col2 = st.columns([1, 3])
            with col1:
                retry_button = st.button(
                    "🔄 Retry Agent",
                    key=f"retry_btn_{agent_name}",
                    type="primary",
                    use_container_width=True
                )

            if retry_button:
                try:
                    # Parse edited input
                    if edited_input.strip():
                        modified_input = json.loads(edited_input)
                    else:
                        modified_input = None

                    # Set running state
                    SessionStateManager.set_running(True)

                    # Progress UI
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(message, progress, error=False):
                        progress_bar.progress(min(progress, 1.0))
                        if error:
                            status_text.error(message)
                        else:
                            status_text.info(message)

                    # Retry agent
                    updated_session = pipeline.retry_agent(
                        session=session,
                        agent_name=agent_name,
                        modified_input=modified_input,
                        progress_callback=progress_callback
                    )

                    # Update session
                    SessionStateManager.set_active_session(updated_session)
                    SessionStateManager.set_running(False)

                    st.success(f"✅ {agent_name} retry completed successfully!")
                    st.rerun()

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {str(e)}")
                    SessionStateManager.set_running(False)
                except Exception as e:
                    st.error(f"Retry failed: {str(e)}")
                    SessionStateManager.set_running(False)


def display_session_results(session: SessionState):
    """Display session results in tabs."""
    st.subheader("Results")

    # Get session directory for image paths (Phase 2)
    pipeline = SessionStateManager.get_pipeline()
    session_dir = Path(pipeline.session_manager.base_directory) / session.session_id

    # Create tabs for each agent (Phase 1 + Phase 2 + Phase 3)
    tabs = st.tabs([
        "Agent 1: Screenplay",
        "Agent 2: Scenes",
        "Agent 3: Shots",
        "Agent 4: Grouping",
        "Agent 5: Characters",
        "Agent 6: Parent Shots",
        "Agent 7: Verification",
        "Agent 8: Child Shots",
        "Agent 9: Verification",
        "Agent 10: Videos",
        "📦 Complete Export"
    ])

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

                # Retry UI
                render_retry_ui(session, "agent_1", agent_output)
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

                # Retry UI
                render_retry_ui(session, "agent_2", agent_output)
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

                # Retry UI
                render_retry_ui(session, "agent_3", agent_output)
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
                        width="stretch"
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
                            width="stretch"
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

                # Retry UI
                render_retry_ui(session, "agent_4", agent_output)
        else:
            st.info("Agent 4 not yet executed")

    # Agent 5 Output - Character Creator
    with tabs[4]:
        if "agent_5" in session.agents:
            agent_output = session.agents["agent_5"]
            if agent_output.status == "completed":
                st.markdown("### Character Creator")

                char_data = agent_output.output_data
                total_chars = char_data.get("total_characters", 0)
                total_grids = char_data.get("total_grids", 0)

                st.info(f"Characters: {total_chars} | Grids: {total_grids}")

                # Display characters with images
                st.markdown("#### Generated Characters")
                for char in char_data.get("characters", []):
                    with st.expander(f"🧑 {char.get('name', 'Unknown')}"):
                        img_path = session_dir / char.get("image_path", "")
                        if img_path.exists():
                            st.image(str(img_path), caption=char.get("name"), width=300)
                        st.write(char.get("description", ""))

                # Display grids
                st.markdown("#### Character Grids")
                for grid in char_data.get("character_grids", []):
                    grid_id = grid.get("grid_id", "Unknown")
                    with st.expander(f"📸 Grid: {grid_id}"):
                        img_path = session_dir / grid.get("grid_path", "")
                        if img_path.exists():
                            st.image(str(img_path), caption=grid_id, width="stretch")
                        st.write(f"Characters: {', '.join(grid.get('characters', []))}")

                # Download
                st.download_button(
                    "📥 Download Character Data",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_characters.json",
                    mime="application/json"
                )
            else:
                st.warning(f"Agent 5 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)

                # Retry UI
                render_retry_ui(session, "agent_5", agent_output)
        else:
            st.info("Agent 5 not yet executed (Phase 2)")

    # Agent 6/7 Output - Parent Shots & Verification (combined)
    with tabs[5]:
        if "agent_7" in session.agents:
            agent_output = session.agents["agent_7"]
            if agent_output.status == "completed":
                st.markdown("### Parent Shots (Verified)")

                parent_data = agent_output.output_data
                total_verified = parent_data.get("metadata", {}).get("total_verified", 0)
                total_soft_fail = parent_data.get("metadata", {}).get("total_soft_failures", 0)

                st.info(f"✓ Verified: {total_verified} | ⚠️ Soft Failures: {total_soft_fail}")

                # Display parent shots
                for parent in parent_data.get("parent_shots", []):
                    shot_id = parent.get("shot_id")
                    status = parent.get("verification_status", "unknown")
                    status_emoji = "✓" if status == "verified" else "⚠️"

                    with st.expander(f"{status_emoji} {shot_id} ({status})"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            img_path = session_dir / parent.get("image_path", "")
                            if img_path.exists():
                                st.image(str(img_path), caption=shot_id, use_container_width=True)

                        with col2:
                            # Video generation button
                            if st.button(
                                "🎬 Generate Video",
                                key=f"gen_video_parent_{shot_id}",
                                use_container_width=True,
                                help="Generate video for this shot"
                            ):
                                try:
                                    pipeline = SessionStateManager.get_pipeline()

                                    with st.spinner(f"Generating video for {shot_id}..."):
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()

                                        def progress_callback(message, progress, error=False):
                                            progress_bar.progress(min(progress, 1.0))
                                            if error:
                                                status_text.error(message)
                                            else:
                                                status_text.info(message)

                                        video_data = pipeline.generate_single_shot_video(
                                            session=session,
                                            shot_id=shot_id,
                                            shot_type="parent",
                                            progress_callback=progress_callback
                                        )

                                        if video_data.get("status") != "failed":
                                            st.success(f"✅ Video generated for {shot_id}!")
                                            if video_data.get("video_url"):
                                                st.video(video_data["video_url"])
                                        else:
                                            st.error(f"Failed: {video_data.get('error')}")

                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                        # Verification details
                        final_ver = parent.get("final_verification", {})
                        if final_ver:
                            st.write(f"**Confidence:** {final_ver.get('confidence', 0):.2f}")
                            if final_ver.get("issues"):
                                # Handle categorized issues (List[Dict] with category and description)
                                issue_strs = [
                                    f"{iss.get('category', 'Issue')}: {iss.get('description', '')}"
                                    for iss in final_ver['issues']
                                ]
                                st.warning(f"**Issues:** {', '.join(issue_strs)}")

                # Download
                st.download_button(
                    "📥 Download Parent Shots Data",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_parent_shots.json",
                    mime="application/json"
                )
            else:
                st.warning(f"Agent 7 status: {agent_output.status}")

                # Retry UI for Agent 7
                render_retry_ui(session, "agent_7", agent_output)
        else:
            st.info("Agent 6/7 not yet executed (Phase 2)")

    # Agent 7 Tab (Verification Details)
    with tabs[6]:
        if "agent_7" in session.agents:
            st.markdown("### Parent Verification Details")
            agent_output = session.agents["agent_7"]
            if agent_output.status == "completed":
                st.json(agent_output.output_data)
            else:
                st.warning(f"Status: {agent_output.status}")
        else:
            st.info("Agent 7 not yet executed")

    # Agent 8/9 Output - Child Shots & Verification (combined)
    with tabs[7]:
        if "agent_9" in session.agents:
            agent_output = session.agents["agent_9"]
            if agent_output.status == "completed":
                st.markdown("### Child Shots (Verified)")

                child_data = agent_output.output_data
                total_verified = child_data.get("metadata", {}).get("total_verified", 0)
                total_soft_fail = child_data.get("metadata", {}).get("total_soft_failures", 0)

                st.info(f"✓ Verified: {total_verified} | ⚠️ Soft Failures: {total_soft_fail}")

                # Display child shots
                child_shots = child_data.get("child_shots", [])
                if len(child_shots) > 20:
                    st.warning(f"Showing first 20 of {len(child_shots)} child shots (download JSON for complete list)")

                for child in child_shots[:20]:  # Limit to first 20 for UI
                    shot_id = child.get("shot_id")
                    status = child.get("verification_status", "unknown")
                    status_emoji = "✓" if status == "verified" else "⚠️"

                    with st.expander(f"{status_emoji} {shot_id} ({status})"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            img_path = session_dir / child.get("image_path", "")
                            if img_path.exists():
                                st.image(str(img_path), caption=shot_id, use_container_width=True)

                        with col2:
                            # Video generation button
                            if st.button(
                                "🎬 Generate Video",
                                key=f"gen_video_child_{shot_id}",
                                use_container_width=True,
                                help="Generate video for this shot"
                            ):
                                try:
                                    pipeline = SessionStateManager.get_pipeline()

                                    with st.spinner(f"Generating video for {shot_id}..."):
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()

                                        def progress_callback(message, progress, error=False):
                                            progress_bar.progress(min(progress, 1.0))
                                            if error:
                                                status_text.error(message)
                                            else:
                                                status_text.info(message)

                                        video_data = pipeline.generate_single_shot_video(
                                            session=session,
                                            shot_id=shot_id,
                                            shot_type="child",
                                            progress_callback=progress_callback
                                        )

                                        if video_data.get("status") != "failed":
                                            st.success(f"✅ Video generated for {shot_id}!")
                                            if video_data.get("video_url"):
                                                st.video(video_data["video_url"])
                                        else:
                                            st.error(f"Failed: {video_data.get('error')}")

                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                # Download
                st.download_button(
                    "📥 Download Child Shots Data",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_child_shots.json",
                    mime="application/json"
                )
            else:
                st.warning(f"Agent 9 status: {agent_output.status}")

                # Retry UI for Agent 9
                render_retry_ui(session, "agent_9", agent_output)
        else:
            st.info("Agent 8/9 not yet executed (Phase 2)")

    # Agent 9 Tab (Verification Details)
    with tabs[8]:
        if "agent_9" in session.agents:
            st.markdown("### Child Verification Details")
            agent_output = session.agents["agent_9"]
            if agent_output.status == "completed":
                st.json(agent_output.output_data)
            else:
                st.warning(f"Status: {agent_output.status}")
        else:
            st.info("Agent 9 not yet executed")

    # Agent 10 Output - Video Dialogue Generator
    with tabs[9]:
        if "agent_10" in session.agents:
            agent_output = session.agents["agent_10"]
            if agent_output.status == "completed":
                st.markdown("### Generated Videos")

                video_data = agent_output.output_data
                total_videos = video_data.get("total_videos", 0)
                successful = video_data.get("successful_videos", 0)
                failed = video_data.get("failed_videos", 0)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Videos", total_videos)
                with col2:
                    st.metric("✓ Successful", successful)
                with col3:
                    st.metric("✗ Failed", failed)

                st.divider()

                # Display videos
                videos = video_data.get("videos", [])

                # Filter by type
                filter_type = st.radio("Filter by type:", ["All", "Parent Shots", "Child Shots"], horizontal=True)

                if filter_type == "Parent Shots":
                    videos = [v for v in videos if v.get("shot_type") == "parent"]
                elif filter_type == "Child Shots":
                    videos = [v for v in videos if v.get("shot_type") == "child"]

                # Display videos
                for video in videos:
                    shot_id = video.get("shot_id")
                    shot_type = video.get("shot_type", "unknown")
                    status = video.get("status", "unknown")

                    status_emoji = "✓" if status == "success" else "✗"
                    type_emoji = "📹" if shot_type == "parent" else "🎬"

                    with st.expander(f"{type_emoji} {status_emoji} {shot_id} ({shot_type})"):
                        if status == "success":
                            # Video URL
                            video_url = video.get("video_url", "")
                            if video_url:
                                st.video(video_url)
                                st.write(f"**Video URL:** {video_url}")

                            # Video details
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Duration:** {video.get('duration_seconds', 'N/A')} seconds")
                                st.write(f"**Generated:** {video.get('generated_at', 'N/A')}")
                            with col2:
                                st.write(f"**FAL Request ID:** {video.get('fal_request_id', 'N/A')}")

                            # Production brief
                            st.write("**Video Prompt:**")
                            st.text(video.get('video_prompt', 'N/A'))

                            # Link to brief file
                            brief_path = session_dir / video.get("production_brief_path", "")
                            if brief_path.exists():
                                with open(brief_path, 'r', encoding='utf-8') as f:
                                    brief_content = f.read()
                                # Determine language based on file extension
                                lang = "json" if brief_path.suffix == ".json" else "yaml"
                                with st.expander("📄 View Full Production Brief"):
                                    st.code(brief_content, language=lang)
                        else:
                            # Failed video
                            st.error(f"**Error:** {video.get('error', 'Unknown error')}")

                # Download button
                st.download_button(
                    "📥 Download Video Metadata",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_videos.json",
                    mime="application/json"
                )
            else:
                st.warning(f"Agent 10 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)

                # Retry UI
                render_retry_ui(session, "agent_10", agent_output)
        else:
            st.info("Agent 10 not yet executed (Phase 3)")

    # Complete Export Tab
    with tabs[10]:
        st.markdown("### 📦 Complete Export")
        st.markdown("Export your Story Architect project in various formats.")

        # Check if Phase 2 completed
        has_phase2 = "agent_9" in session.agents and session.agents["agent_9"].status == "completed"

        if has_phase2:
            st.success("✓ Phase 2 complete - All images generated!")
        else:
            st.info("Phase 2 in progress or not started - Exports will include current outputs")

        # Export Option 1a: HTML Export (Single File)
        st.markdown("#### Option 1a: HTML Export (Single File)")
        st.markdown("**Single self-contained HTML file** with all images embedded. Best for smaller projects (< 50MB).")

        if st.button("🌐 Export as Single HTML", type="primary", key="html_export"):
            try:
                from core.export import generate_html_export

                # Get session directory from pipeline's session manager
                pipeline = SessionStateManager.get_pipeline()
                session_dir = Path(pipeline.session_manager.base_directory) / session.session_id

                # Generate HTML
                with st.spinner("Generating HTML export with embedded images..."):
                    html_path = generate_html_export(session_dir, session.model_dump())

                st.success(f"✓ HTML export generated: {html_path.name}")
                st.info("💡 Open this file in any browser, or import directly into Notion!")

                # Provide download
                with open(html_path, 'rb') as f:
                    st.download_button(
                        "📥 Download HTML File",
                        data=f.read(),
                        file_name=html_path.name,
                        mime="text/html",
                        width="stretch",
                        key="html_download"
                    )

            except Exception as e:
                st.error(f"Failed to generate HTML export: {str(e)}")

        st.markdown("---")

        # Export Option 1b: HTML Export (Auto-Split Parts) - NEW
        st.markdown("#### Option 1b: HTML Export (Auto-Split Parts) 🆕")
        st.markdown("**Automatically splits into multiple files** if needed. Each part < 50MB. Recommended for large projects.")

        if st.button("🌐 Export as HTML Parts", type="primary", key="html_parts_export"):
            try:
                from core.export import generate_html_export_parts

                # Get session directory from pipeline's session manager
                pipeline = SessionStateManager.get_pipeline()
                session_dir = Path(pipeline.session_manager.base_directory) / session.session_id

                # Generate parts
                with st.spinner("Generating HTML export with auto-splitting..."):
                    html_paths = generate_html_export_parts(session_dir, session.model_dump())

                # Success message
                if len(html_paths) == 1:
                    st.success(f"✓ Generated 1 file (project fits in single file!)")
                else:
                    st.success(f"✓ Generated {len(html_paths)} parts (split to meet 50MB limit)")

                st.info("💡 Each file can be imported to Notion separately. Open in any browser!")

                # Provide download button for each part
                for idx, html_path in enumerate(html_paths, 1):
                    file_size_mb = html_path.stat().st_size / (1024 * 1024)
                    with open(html_path, 'rb') as f:
                        st.download_button(
                            f"📥 Download Part {idx} of {len(html_paths)} ({file_size_mb:.1f} MB)",
                            data=f.read(),
                            file_name=html_path.name,
                            mime="text/html",
                            width="stretch",
                            key=f"html_part_download_{idx}"
                        )

            except Exception as e:
                st.error(f"Failed to generate HTML parts: {str(e)}")

        st.markdown("---")

        # Export Option 2: Notion ZIP
        st.markdown("#### Option 2: Notion ZIP Export")
        st.markdown("Clean ZIP with markdown + images folder. For Notion users who prefer ZIP imports.")

        if st.button("📥 Export for Notion (ZIP)", key="notion_export"):
            try:
                from core.export import generate_notion_export_zip

                # Get session directory from pipeline's session manager
                pipeline = SessionStateManager.get_pipeline()
                session_dir = Path(pipeline.session_manager.base_directory) / session.session_id

                # Generate Notion ZIP
                with st.spinner("Generating Notion export ZIP..."):
                    zip_path = generate_notion_export_zip(session_dir, session.model_dump())

                st.success(f"✓ Notion export generated: {zip_path.name}")

                # Provide download
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Notion Export",
                        data=f.read(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        width="stretch",
                        key="notion_download"
                    )

            except Exception as e:
                st.error(f"Failed to generate Notion export: {str(e)}")

        st.markdown("---")

        # Export Option 3: Complete Archive
        st.markdown("#### Option 3: Complete Archive")
        st.markdown("Full ZIP export including all JSON files, markdown, and images. For archival/debugging purposes.")

        if st.button("📦 Export Complete Archive", key="complete_export"):
            try:
                from core.export import generate_complete_export_zip

                # Get session directory from pipeline's session manager
                pipeline = SessionStateManager.get_pipeline()
                session_dir = Path(pipeline.session_manager.base_directory) / session.session_id

                # Generate complete ZIP
                with st.spinner("Generating complete export ZIP..."):
                    zip_path = generate_complete_export_zip(session_dir, session.model_dump())

                st.success(f"✓ Complete export generated: {zip_path.name}")

                # Provide download
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Complete Archive",
                        data=f.read(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        width="stretch",
                        key="complete_download"
                    )

            except Exception as e:
                st.error(f"Failed to generate complete export: {str(e)}")


def render_resume_session_page():
    """Render the resume session page."""
    st.header("Resume Session")

    # Load recent sessions
    pipeline = SessionStateManager.get_pipeline()
    sessions = pipeline.list_sessions(limit=10)

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
        session = pipeline.get_session(session_id)

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
                options=["agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6", "agent_7", "agent_8", "agent_9", "agent_10"]
            )

            if st.button("Resume Pipeline", type="primary"):
                try:
                    SessionStateManager.set_active_session(session)
                    SessionStateManager.set_running(True)

                    # Resume pipeline
                    updated_session = pipeline.resume_from_agent(
                        session=session,
                        agent_name=resume_from
                    )

                    SessionStateManager.set_active_session(updated_session)
                    SessionStateManager.set_running(False)

                    st.success("Pipeline resumed and completed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to resume pipeline: {str(e)}")
                    SessionStateManager.set_running(False)


def render_session_history_page():
    """Render the session history page."""
    st.header("Session History")

    # Load all sessions
    pipeline = SessionStateManager.get_pipeline()
    sessions = pipeline.list_sessions(limit=50)

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
                    loaded_session = pipeline.get_session(session['session_id'])
                    if loaded_session:
                        SessionStateManager.set_active_session(loaded_session)
                        display_session_results(loaded_session)

            with col2:
                if st.button(f"Delete Session", key=f"delete_{session['session_id']}", type="secondary"):
                    if pipeline.delete_session(session['session_id']):
                        st.success("Session deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete session")


if __name__ == "__main__":
    main()
