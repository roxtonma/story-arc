"""
Export Utilities
Functions for exporting pipeline outputs to various formats.
"""

from typing import Dict, Any, List
from datetime import datetime
import html


def generate_notion_markdown(
    shot_grouping_data: Dict[str, Any],
    shot_breakdown_data: Dict[str, Any]
) -> str:
    """
    Convert Agent 4 shot grouping JSON to Notion-compatible Markdown.
    Uses HTML <details> tags for collapsible parent/child hierarchy.

    Args:
        shot_grouping_data: Dictionary containing ShotGrouping JSON output from Agent 4
        shot_breakdown_data: Dictionary containing ShotBreakdown JSON output from Agent 3

    Returns:
        Formatted Markdown string with ALL data preserved (no truncation)
    """
    lines = []

    # Create shot lookup dictionary from Agent 3 output
    shots_by_id = {}
    for shot in shot_breakdown_data.get('shots', []):
        shot_id = shot.get('shot_id')
        if shot_id:
            shots_by_id[shot_id] = shot

    # Header
    lines.append("# Shot Grouping Export\n")

    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Parent Shots:** {shot_grouping_data.get('total_parent_shots', 0)}\n")
    lines.append(f"**Total Child Shots:** {shot_grouping_data.get('total_child_shots', 0)}\n")

    strategy = shot_grouping_data.get('grouping_strategy', 'N/A')
    lines.append(f"\n**Grouping Strategy:** {strategy}\n")

    lines.append("\n---\n")

    # Process parent shots grouped by scene
    parent_shots = shot_grouping_data.get('parent_shots', [])

    # Group by scene_id for better organization
    shots_by_scene = {}
    for shot in parent_shots:
        scene_id = shot.get('scene_id', 'UNKNOWN')
        if scene_id not in shots_by_scene:
            shots_by_scene[scene_id] = []
        shots_by_scene[scene_id].append(shot)

    # Render each scene
    for scene_id, shots in sorted(shots_by_scene.items()):
        lines.append(f"\n## Scene: {scene_id}\n")

        for shot in shots:
            lines.append(_format_grouped_shot(shot, shots_by_id, level=0))
            lines.append("\n---\n")

    return "\n".join(lines)


def _format_grouped_shot(
    shot: Dict[str, Any],
    shots_by_id: Dict[str, Dict[str, Any]],
    level: int = 0
) -> str:
    """
    Recursively format a grouped shot with its children.
    Uses HTML <details> tags for collapsible sections.

    Args:
        shot: GroupedShot dictionary (from Agent 4)
        shots_by_id: Dictionary mapping shot_id to full Shot data (from Agent 3)
        level: Nesting level (for indentation)

    Returns:
        Markdown formatted string with complete shot data
    """
    indent = "  " * level  # 2 spaces per level
    lines = []

    # Get shot grouping details (from Agent 4)
    shot_id = shot.get('shot_id', 'UNKNOWN')
    shot_type = shot.get('shot_type', 'unknown')
    grouping_reason = shot.get('grouping_reason', 'N/A')
    child_shots = shot.get('child_shots', [])

    # Look up full shot data from Agent 3 output
    shot_data = shots_by_id.get(shot_id, {})
    shot_description = shot_data.get('shot_description', 'No description')

    # Start details block
    # Summary shows shot ID and full description (no truncation)
    summary = f"{shot_id} - {shot_description}"
    lines.append(f"{indent}<details>")
    lines.append(f"{indent}<summary><strong>{_escape_html(summary)}</strong></summary>")
    lines.append("")

    # Shot metadata
    lines.append(f"{indent}### Shot Details\n")
    lines.append(f"{indent}- **Shot ID:** {shot_id}")
    lines.append(f"{indent}- **Shot Type:** {shot_type.capitalize()}")

    parent_shot_id = shot.get('parent_shot_id')
    if parent_shot_id:
        lines.append(f"{indent}- **Parent Shot:** {parent_shot_id}")

    scene_id = shot.get('scene_id', 'UNKNOWN')
    lines.append(f"{indent}- **Scene:** {scene_id}")

    # Shot data fields
    location = shot_data.get('location', 'N/A')
    lines.append(f"{indent}- **Location:** {location}")

    characters = shot_data.get('characters', [])
    if characters:
        char_list = ", ".join(characters)
        lines.append(f"{indent}- **Characters:** {char_list}")
    else:
        lines.append(f"{indent}- **Characters:** None")

    dialogue = shot_data.get('dialogue')
    if dialogue:
        lines.append(f"{indent}- **Dialogue:** \"{_escape_html(dialogue)}\"")
    else:
        lines.append(f"{indent}- **Dialogue:** None")

    # First Frame (complete, no truncation)
    first_frame = shot_data.get('first_frame', 'N/A')
    lines.append(f"\n{indent}#### First Frame Description\n")
    lines.append(f"{indent}{first_frame}\n")

    # Animation (complete, no truncation)
    animation = shot_data.get('animation', 'N/A')
    lines.append(f"{indent}#### Animation Instructions\n")
    lines.append(f"{indent}{animation}\n")

    # Grouping Reason
    lines.append(f"{indent}#### Grouping Reason\n")
    lines.append(f"{indent}{grouping_reason}\n")

    # Child Shots (recursive)
    if child_shots:
        lines.append(f"{indent}### Child Shots\n")

        for child in child_shots:
            # Recursively format children (increase indent level, pass shot lookup)
            child_markdown = _format_grouped_shot(child, shots_by_id, level=level + 1)
            lines.append(child_markdown)
            lines.append("")

    # Close details block
    lines.append(f"{indent}</details>")

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """
    Escape HTML special characters to prevent breaking HTML tags.

    Args:
        text: Raw text string

    Returns:
        HTML-escaped text
    """
    return html.escape(str(text))


def generate_scene_breakdown_markdown(scene_breakdown_data: Dict[str, Any]) -> str:
    """
    Convert Agent 2 scene breakdown JSON to readable Markdown.

    Args:
        scene_breakdown_data: Dictionary containing SceneBreakdown JSON output

    Returns:
        Formatted Markdown string
    """
    lines = []

    lines.append("# Scene Breakdown Export\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Scenes:** {scene_breakdown_data.get('total_scenes', 0)}\n")

    lines.append("\n---\n")

    # Process each scene
    scenes = scene_breakdown_data.get('scenes', [])

    for scene in scenes:
        scene_id = scene.get('scene_id', 'UNKNOWN')
        lines.append(f"\n## {scene_id}\n")

        # Location
        location = scene.get('location', {})
        lines.append(f"**Location:** {location.get('name', 'N/A')}\n")
        lines.append(f"{location.get('description', 'N/A')}\n")

        # Time of day
        time_of_day = scene.get('time_of_day')
        if time_of_day:
            lines.append(f"\n**Time of Day:** {time_of_day}\n")

        # Characters
        characters = scene.get('characters', [])
        if characters:
            lines.append(f"\n### Characters\n")
            for char in characters:
                char_name = char.get('name', 'Unknown')
                char_desc = char.get('description', 'N/A')
                lines.append(f"\n**{char_name}**\n{char_desc}\n")

        # Screenplay Text
        screenplay_text = scene.get('screenplay_text', 'N/A')
        lines.append(f"\n### Screenplay\n")
        lines.append(f"```\n{screenplay_text}\n```\n")

        # Subscenes
        subscenes = scene.get('subscenes', [])
        if subscenes:
            lines.append(f"\n### Subscenes\n")
            for subscene in subscenes:
                subscene_id = subscene.get('subscene_id', 'UNKNOWN')
                event = subscene.get('event', 'UNKNOWN')
                lines.append(f"\n**{subscene_id}** - {event}\n")

                if event == "CHARACTER_ADDED":
                    char_added = subscene.get('character_added', {})
                    if char_added:
                        lines.append(f"- **Character:** {char_added.get('name', 'Unknown')}\n")
                        lines.append(f"- **Description:** {char_added.get('description', 'N/A')}\n")

                excerpt = subscene.get('screenplay_excerpt', '')
                if excerpt:
                    lines.append(f"```\n{excerpt}\n```\n")

        lines.append("\n---\n")

    return "\n".join(lines)


def generate_shot_breakdown_markdown(shot_breakdown_data: Dict[str, Any]) -> str:
    """
    Convert Agent 3 shot breakdown JSON to readable Markdown.

    Args:
        shot_breakdown_data: Dictionary containing ShotBreakdown JSON output

    Returns:
        Formatted Markdown string
    """
    lines = []

    lines.append("# Shot Breakdown Export\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Shots:** {shot_breakdown_data.get('total_shots', 0)}\n")

    lines.append("\n---\n")

    # Group shots by scene
    shots = shot_breakdown_data.get('shots', [])
    shots_by_scene = {}

    for shot in shots:
        scene_id = shot.get('scene_id', 'UNKNOWN')
        if scene_id not in shots_by_scene:
            shots_by_scene[scene_id] = []
        shots_by_scene[scene_id].append(shot)

    # Render each scene's shots
    for scene_id, scene_shots in sorted(shots_by_scene.items()):
        lines.append(f"\n## Scene: {scene_id}\n")

        for shot in scene_shots:
            shot_id = shot.get('shot_id', 'UNKNOWN')
            shot_desc = shot.get('shot_description', 'No description')

            lines.append(f"\n### {shot_id}\n")
            lines.append(f"{shot_desc}\n")

            # Location and Characters
            location = shot.get('location', 'N/A')
            characters = shot.get('characters', [])
            char_list = ", ".join(characters) if characters else "None"

            lines.append(f"- **Location:** {location}")
            lines.append(f"- **Characters:** {char_list}")

            # Dialogue
            dialogue = shot.get('dialogue')
            if dialogue:
                lines.append(f"- **Dialogue:** \"{dialogue}\"")
            else:
                lines.append(f"- **Dialogue:** None")

            # First Frame
            first_frame = shot.get('first_frame', 'N/A')
            lines.append(f"\n**First Frame:**\n{first_frame}\n")

            # Animation
            animation = shot.get('animation', 'N/A')
            lines.append(f"\n**Animation:**\n{animation}\n")

            lines.append("\n---\n")

    return "\n".join(lines)
