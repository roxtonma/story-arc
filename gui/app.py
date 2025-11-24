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
            width="stretch"
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

    # Get session directory for image paths (Phase 2)
    session_manager = st.session_state.pipeline.session_manager
    session_dir = Path(session_manager.base_directory) / session.session_id

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
        "Agent 11: Final Edit",
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
                        img_path = session_dir / parent.get("image_path", "")
                        if img_path.exists():
                            st.image(str(img_path), caption=shot_id, width="stretch")

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
                        img_path = session_dir / child.get("image_path", "")
                        if img_path.exists():
                            st.image(str(img_path), caption=shot_id, width="stretch")

                # Download
                st.download_button(
                    "📥 Download Child Shots Data",
                    data=json.dumps(agent_output.output_data, indent=2),
                    file_name=f"{session.session_id}_child_shots.json",
                    mime="application/json"
                )
            else:
                st.warning(f"Agent 9 status: {agent_output.status}")
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
        else:
            st.info("Agent 10 not yet executed (Phase 3)")

    # Agent 11: Final Edit
    with tabs[10]:
        if "agent_11" in session.agents:
            agent_output = session.agents["agent_11"]
            if agent_output.status == "completed":
                st.markdown("### 🎬 Final Edited Video")

                edit_data = agent_output.output_data
                total_duration = edit_data.get("total_duration", 0)
                edit_metadata = edit_data.get("edit_metadata", {})

                # Stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                with col2:
                    st.metric("Scenes", edit_metadata.get("scenes_edited", 0))
                with col3:
                    st.metric("Total Shots", edit_metadata.get("total_shots", 0))
                with col4:
                    editing_method = edit_metadata.get("editing_method", "unknown")
                    method_label = "✨ AI Edit" if editing_method == "gemini_edl" else "🔧 Heuristic"
                    st.metric("Method", method_label)

                st.divider()

                # Master Video
                st.markdown("#### 🎥 Master Video (All Scenes)")
                master_video_path = session_dir / edit_data.get("master_video_path", "")
                if master_video_path.exists():
                    st.video(str(master_video_path))
                    st.caption(f"Duration: {total_duration:.2f} seconds")

                    # Download master video
                    with open(master_video_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Master Video",
                            data=f,
                            file_name=f"{session.session_id}_master_final.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    st.error(f"Master video not found: {edit_data.get('master_video_path')}")

                st.divider()

                # Scene Videos
                st.markdown("#### 📽️ Scene Videos")
                scene_videos = edit_data.get("scene_videos", [])

                if scene_videos:
                    for scene in scene_videos:
                        scene_id = scene.get("scene_id")
                        duration = scene.get("duration", 0)
                        shot_count = scene.get("shot_count", 0)

                        with st.expander(f"🎬 {scene_id} - {duration:.1f}s ({shot_count} shots)"):
                            scene_video_path = session_dir / scene.get("video_path", "")
                            if scene_video_path.exists():
                                st.video(str(scene_video_path))
                                st.caption(f"Duration: {duration:.2f}s | Shots: {shot_count}")

                                # Download scene video
                                with open(scene_video_path, 'rb') as f:
                                    st.download_button(
                                        f"📥 Download {scene_id}",
                                        data=f,
                                        file_name=f"{session.session_id}_{scene_id}.mp4",
                                        mime="video/mp4",
                                        key=f"download_{scene_id}"
                                    )
                            else:
                                st.error(f"Scene video not found: {scene.get('video_path')}")
                else:
                    st.info("No scene videos available")

                st.divider()

                # Edit Timeline Info
                st.markdown("#### 📋 Edit Decision List")
                edit_timeline = edit_data.get("edit_timeline", {})

                with st.expander("View Edit Timeline Details"):
                    st.json(edit_timeline)

                # Download EDL
                st.download_button(
                    "📥 Download Edit Data (JSON)",
                    data=json.dumps(edit_data, indent=2),
                    file_name=f"{session.session_id}_edit_output.json",
                    mime="application/json"
                )

            else:
                st.warning(f"Agent 11 status: {agent_output.status}")
                if agent_output.error_message:
                    st.error(agent_output.error_message)
        else:
            st.info("Agent 11 not yet executed - Run Agent 10 first to generate videos")

    # Complete Export Tab
    with tabs[11]:
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
                session_manager = st.session_state.pipeline.session_manager
                session_dir = Path(session_manager.base_directory) / session.session_id

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
                session_manager = st.session_state.pipeline.session_manager
                session_dir = Path(session_manager.base_directory) / session.session_id

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
                session_manager = st.session_state.pipeline.session_manager
                session_dir = Path(session_manager.base_directory) / session.session_id

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
                session_manager = st.session_state.pipeline.session_manager
                session_dir = Path(session_manager.base_directory) / session.session_id

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
                options=["agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6", "agent_7", "agent_8", "agent_9", "agent_10", "agent_11"]
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
