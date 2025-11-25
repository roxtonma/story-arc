"""
Story Architect - CLI Entry Point
Command-line interface for running the pipeline without GUI.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from core.pipeline import Pipeline


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Story Architect - AI-Powered Script-to-Shot Conversion"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the pipeline')
    run_parser.add_argument(
        'input_file',
        type=str,
        help='Path to input file (logline or screenplay)'
    )
    run_parser.add_argument(
        '--start-agent',
        choices=['agent_1', 'agent_2'],
        default='agent_1',
        help='Starting agent (default: agent_1)'
    )
    run_parser.add_argument(
        '--name',
        type=str,
        help='Session name (optional)'
    )
    run_parser.add_argument(
        '--output',
        type=str,
        help='Output directory (optional, overrides config)'
    )

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume a session')
    resume_parser.add_argument(
        'session_id',
        type=str,
        help='Session ID to resume'
    )
    resume_parser.add_argument(
        '--from-agent',
        choices=['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7', 'agent_8', 'agent_9', 'agent_10', 'agent_11'],
        default='agent_2',
        help='Agent to resume from (default: agent_2)'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List sessions')
    list_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of sessions to list (default: 20)'
    )

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch the GUI')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if not args.command:
        parser.print_help()
        return

    # Initialize pipeline
    try:
        pipeline = Pipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        sys.exit(1)

    # Execute command
    if args.command == 'run':
        run_pipeline(pipeline, args)
    elif args.command == 'resume':
        resume_session(pipeline, args)
    elif args.command == 'list':
        list_sessions(pipeline, args)
    elif args.command == 'gui':
        launch_gui()


def run_pipeline(pipeline: Pipeline, args):
    """Run the pipeline with input file."""
    input_file = Path(args.input_file)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    # Read input
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = f.read()
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        sys.exit(1)

    logger.info(f"Creating session with start agent: {args.start_agent}")

    # Create session
    try:
        session = pipeline.create_session(
            input_data=input_data,
            start_agent=args.start_agent,
            session_name=args.name
        )

        logger.info(f"Session created: {session.session_id}")

        # Run pipeline
        logger.info("Running pipeline...")

        def progress_callback(message, progress, error=False):
            if error:
                logger.error(message)
            else:
                logger.info(f"[{int(progress * 100)}%] {message}")

        updated_session = pipeline.run_pipeline(
            session=session,
            progress_callback=progress_callback
        )

        logger.success(f"Pipeline completed successfully!")
        logger.info(f"Session ID: {updated_session.session_id}")
        logger.info(f"Output directory: outputs/projects/{updated_session.session_id}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


def resume_session(pipeline: Pipeline, args):
    """Resume a session from specified agent."""
    logger.info(f"Loading session: {args.session_id}")

    # Load session
    session = pipeline.get_session(args.session_id)

    if not session:
        logger.error(f"Session not found: {args.session_id}")
        sys.exit(1)

    logger.info(f"Resuming from {args.from_agent}...")

    # Resume pipeline
    try:
        def progress_callback(message, progress, error=False):
            if error:
                logger.error(message)
            else:
                logger.info(f"[{int(progress * 100)}%] {message}")

        updated_session = pipeline.resume_from_agent(
            session=session,
            agent_name=args.from_agent,
            progress_callback=progress_callback
        )

        logger.success("Pipeline resumed and completed successfully!")
        logger.info(f"Output directory: outputs/projects/{updated_session.session_id}")

    except Exception as e:
        logger.error(f"Failed to resume pipeline: {str(e)}")
        sys.exit(1)


def list_sessions(pipeline: Pipeline, args):
    """List available sessions."""
    sessions = pipeline.list_sessions(limit=args.limit)

    if not sessions:
        logger.info("No sessions found.")
        return

    logger.info(f"Found {len(sessions)} session(s):")
    print()

    for session in sessions:
        print(f"📁 {session['session_name']}")
        print(f"   ID: {session['session_id']}")
        print(f"   Status: {session['status']}")
        print(f"   Current Agent: {session['current_agent']}")
        print(f"   Created: {session['created_at']}")
        print(f"   Updated: {session['updated_at']}")
        print()


def launch_gui():
    """Launch the Streamlit GUI."""
    import subprocess

    logger.info("Launching GUI...")

    try:
        subprocess.run(
            ["streamlit", "run", "gui/app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch GUI: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Streamlit not found. Install it with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
