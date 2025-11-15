# Story Architect

**AI-Powered Multi-Agent Pipeline for Script-to-Shot-to-Image Conversion**

Story Architect is a production-ready system that transforms story concepts into fully generated cinematic images. Powered by Google Gemini 2.5 Pro and Gemini 2.5 Flash Image, it uses a specialized nine-agent pipeline to generate screenplays, break them into shots, and produce consistent character-driven images ready for AI video generation.

## Features

### Phase 1: Script-to-Shot Conversion
- **Screenplay Generation**: Transform loglines into dialogue-driven screenplays
- **Scene Breakdown**: Verbose location and character descriptions
- **Shot Planning**: Detailed shot descriptions with first-frame and animation prompts
- **Shot Grouping**: Parent/child relationships for optimized generation

### Phase 2: Image Generation & Verification (NEW)
- **Character Creation**: Generate consistent 1024x1024 character portraits with grid layouts
- **Parent Shot Images**: Transform character grids into cinematic parent shots
- **Child Shot Images**: Generate consistent child shots using parent references
- **AI Verification**: Automated quality assessment with retry logic
- **Multi-Format Export**: HTML (single/multi-part), Notion ZIP, complete archives

### Core Features
- **Flexible Entry Points**: Start from a logline or bring your own screenplay
- **Session Management**: Save progress and resume from any agent in the pipeline
- **Smart Error Handling**: Automatic retry with feedback loop (up to 3 attempts) plus manual intervention
- **Modular Design**: Each agent can be customized independently via prompt templates
- **Interactive GUI**: Clean Streamlit interface with image preview and export
- **CLI Support**: Full command-line interface for automation

## Architecture

### Nine-Agent Pipeline

**Phase 1: Script-to-Shot (Agents 1-4)**

**Agent 1: Screenplay Generator**
- Input: Logline, story concept, or rough script
- Output: Dialogue-driven screenplay with proper formatting
- Format: Plain text

**Agent 2: Scene Breakdown**
- Input: Screenplay
- Output: Scenes with verbose location and character descriptions
- Features: Hybrid subscene structure with character entrance tracking
- Format: JSON

**Agent 3: Shot Breakdown**
- Input: Scene breakdown
- Output: Individual shots with three components:
  - Shot Description: What happens in the shot
  - First Frame: Verbose visual description for image generation
  - Animation: Instructions for animating the first frame
- Format: Strict JSON schema with validation

**Agent 4: Shot Grouping**
- Input: Shot breakdown
- Output: Parent/child shot relationships in nested hierarchy
- Purpose: Optimize generation by grouping similar shots
- Features: Cross-scene grouping, multi-level nesting
- Format: Nested JSON

**Phase 2: Image Generation & Verification (Agents 5-9)**

**Agent 5: Character Creation**
- Input: Scene breakdown (Agent 2 output)
- Output: 1024x1024 character portraits + combination grids
- Model: Gemini 2.5 Flash Image generation
- Features: Consistent character design, grid layouts for references
- Storage: Images saved to session directory

**Agent 6: Parent Shot Image Generation**
- Input: Agent 4 output (parent shots) + character grids
- Output: Generated parent shot images (1024x1024 or 1280x768)
- Features: Uses character grids as visual references
- Model: Gemini 2.5 Flash Image generation

**Agent 7: Parent Shot Verification**
- Input: Generated parent shot images + shot metadata
- Output: Verification results with quality scores
- Features: Multimodal AI review, retry logic, soft failure mode
- Model: Gemini 2.5 Pro vision analysis

**Agent 8: Child Shot Image Generation**
- Input: Agent 4 output (child shots) + parent shot images + character grids
- Output: Generated child shot images
- Features: Uses parent shots and character grids for consistency
- Model: Gemini 2.5 Flash Image generation

**Agent 9: Child Shot Verification**
- Input: Generated child shot images + parent references
- Output: Verification results with consistency assessment
- Features: Cross-references parent shots for continuity
- Model: Gemini 2.5 Pro vision analysis

### Tech Stack

- Python 3.10+
- Google Gemini 2.5 Pro (text generation, vision analysis)
- Google Gemini 2.5 Flash Image (image generation)
- Streamlit (GUI)
- Pillow (image manipulation)
- Pydantic (validation)
- YAML configuration + environment variables

## Installation

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key (get one at https://aistudio.google.com/apikey)
- Optional: Google Cloud project with Vertex AI enabled (for enterprise deployments)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ShadoW-Shinigami/story-architect.git
cd story-architect
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_api_key_here
```

5. (Optional) Customize `config.yaml` for agent behavior, output paths, and model settings

## Usage

### Launch GUI

```bash
streamlit run gui/app.py
```

The application opens at `http://localhost:8501`

### GUI Workflow

**New Project:**
1. Select "New Project" from sidebar
2. Choose starting point:
   - **Agent 1**: Start from logline/story concept (full pipeline)
   - **Agent 2**: Start with existing screenplay (skip screenplay generation)
3. Enter your input text
4. Optional: name your project
5. Click "Run Pipeline"
6. Monitor progress across agent tabs (1-9)
7. View generated images in Phase 2 tabs
8. Export results using export buttons

**Resume Session:**
1. Select "Resume Session" from sidebar
2. Choose previous session
3. Review existing outputs and images
4. Select agent to resume from (1-9)
5. Click "Resume Pipeline"

**Session History:**
- View all past sessions
- Browse outputs and generated images
- Delete old sessions

**Export Options:**
- **Single HTML**: All content in one file with embedded images
- **Multi-Part HTML**: Auto-split for files >50MB
- **Notion ZIP**: Markdown + images for Notion import
- **Complete Archive**: All outputs, images, and metadata

### CLI Usage

Run complete pipeline:
```bash
python main.py run input.txt --start-agent agent_1 --name "My Project"
```

Resume from specific agent:
```bash
python main.py resume <session_id> --from-agent agent_5
```

List sessions:
```bash
python main.py list --limit 20
```

Launch GUI from CLI:
```bash
python main.py gui
```

### Output Structure

All outputs saved to `outputs/projects/<session_id>/`:

**Phase 1 Outputs:**
- `session_state.json` - Complete session state
- `agent_1_output.txt` - Generated screenplay
- `agent_2_output.json` - Scene breakdown
- `agent_3_output.json` - Shot breakdown with visual prompts
- `agent_4_output.json` - Grouped shot hierarchy

**Phase 2 Outputs:**
- `agent_5_output.json` - Character data with image paths
- `agent_6_output.json` - Parent shot image paths and metadata
- `agent_7_output.json` - Parent verification results
- `agent_8_output.json` - Child shot image paths and metadata
- `agent_9_output.json` - Child verification results
- `characters/` - Generated character portraits and grids
- `parent_shots/` - Generated parent shot images
- `child_shots/` - Generated child shot images

## Configuration

### API Authentication

**Option 1: Direct API (Recommended for development)**
```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

**Option 2: Vertex AI (Recommended for production)**
```bash
# .env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=google-credentials.json
```

Set `use_vertex_ai: true` in `config.yaml`

### Agent Customization

Edit `config.yaml` to adjust agent behavior:
```yaml
agents:
  agent_1:
    temperature: 0.8          # Creativity (0.0-1.0)
    max_output_tokens: 32000  # Max output length
    enabled: true             # Enable/disable

  agent_5:
    temperature: 0.7
    max_output_tokens: 32000
    enabled: true
```

### Prompt Templates

All prompts are in `prompts/` as `.txt` files:
- Use `{input}` placeholder for input data
- Changes take effect immediately
- No code modification needed

## Project Structure

```
story-architect/
├── agents/                           # Agent implementations
│   ├── base_agent.py                # Abstract base class
│   ├── agent_1_screenplay.py        # Phase 1: Screenplay generation
│   ├── agent_2_scene_breakdown.py   # Phase 1: Scene breakdown
│   ├── agent_3_shot_breakdown.py    # Phase 1: Shot breakdown
│   ├── agent_4_grouping.py          # Phase 1: Shot grouping
│   ├── agent_5_character.py         # Phase 2: Character creation
│   ├── agent_6_parent_generator.py  # Phase 2: Parent shot images
│   ├── agent_7_parent_verification.py # Phase 2: Parent verification
│   ├── agent_8_child_generator.py   # Phase 2: Child shot images
│   └── agent_9_child_verification.py # Phase 2: Child verification
├── core/                             # Core components
│   ├── gemini_client.py             # API wrapper with retry logic
│   ├── pipeline.py                  # Pipeline orchestration (9 agents)
│   ├── session_manager.py           # Session persistence
│   ├── validators.py                # Pydantic schemas for all agents
│   ├── image_utils.py               # Image manipulation utilities
│   └── export.py                    # Multi-format export system
├── gui/                              # Streamlit GUI
│   └── app.py                       # Main app with Phase 1 & 2 tabs
├── prompts/                          # Agent prompt templates
│   ├── agent_1_prompt.txt
│   ├── agent_2_prompt.txt
│   ├── agent_3_prompt.txt
│   ├── agent_4_prompt.txt
│   ├── agent_5_prompt.txt
│   ├── agent_6_prompt.txt
│   ├── agent_7_prompt.txt
│   ├── agent_8_prompt.txt
│   └── agent_9_prompt.txt
├── examples/                         # Example input files
├── config.yaml                       # Configuration (9 agents)
├── .env.example                      # Environment template
├── requirements.txt                  # Python dependencies
└── main.py                           # CLI entry point
```

## Example Workflow

```
1. Input: "A detective discovers a conspiracy in a futuristic city"
   ↓ Agent 1
2. Output: Full dialogue-driven screenplay (15 scenes)
   ↓ Agent 2
3. Output: 15 scenes with verbose character/location descriptions
   ↓ Agent 3
4. Output: 45 shots with first-frame prompts and animations
   ↓ Agent 4
5. Output: 12 parent shots, 33 child shots (hierarchical grouping)
   ↓ Agent 5
6. Output: 8 character portraits + 4 combination grids (1024x1024)
   ↓ Agent 6
7. Output: 12 parent shot images generated (1280x768)
   ↓ Agent 7
8. Output: Verification results (11/12 passed, 1 retry succeeded)
   ↓ Agent 8
9. Output: 33 child shot images generated (consistent with parents)
   ↓ Agent 9
10. Output: Final verification (32/33 passed, 1 soft fail documented)
```

Result: Complete cinematic sequence with 45 generated images ready for video generation.

## Troubleshooting

**API Key Issues**
```
Error: Google API key not found
```
Ensure `.env` file exists with valid `GEMINI_API_KEY`

**Rate Limits**
```
Error: Quota exceeded / Rate limit reached
```
Enable billing at https://aistudio.google.com/api-keys for higher limits (free tier: 15 RPM, paid: 1000+ RPM)

**Import Errors**
```
ModuleNotFoundError: No module named 'google.genai'
```
Activate virtual environment and run `pip install -r requirements.txt`

**Image Generation Errors**
```
Error: Image generation failed
```
- Check API key has image generation permissions
- Verify billing is enabled (image generation requires paid tier)
- Review Agent 5-9 logs for specific error messages

**Agent Validation Failures**
- System auto-retries up to 3 times with feedback
- Check error messages in GUI
- Review intermediate outputs in session history
- Manually edit outputs and resume if needed

**Export File Size Issues**
- Large HTML exports auto-split into multiple parts (>50MB)
- Use Notion ZIP export for better portability
- Use Complete Archive for full backup

## Billing & Rate Limits

The Gemini API tier (free vs paid) is determined by your Google AI Studio billing setup, not by code configuration.

**Free Tier:**
- 15 requests per minute
- Limited features
- **No image generation support**
- Data may be used to improve Google products

**Paid Tier (Required for Phase 2):**
- 1000+ requests per minute (model-dependent)
- **Full image generation access** (Gemini 2.5 Flash Image)
- Context caching, batch API
- Data NOT used for model improvement
- Enable at: https://aistudio.google.com/api-keys

**Note:** Phase 2 (Agents 5-9) requires paid tier for image generation.

See pricing: https://ai.google.dev/gemini-api/docs/pricing

## Known Limitations

- No video generation integration (outputs images/prompts only)
- Single model support (Gemini 2.5 Pro/Flash Image)
- Manual JSON editing required for session modifications
- No batch processing
- Image generation requires paid API tier

## Future Roadmap

- **Video generation integration** (animate generated images)
- **Interactive image editing** (regenerate specific shots)
- Character/location database with consistency tracking
- Batch processing for multiple projects
- Multi-model support (Stable Diffusion, Midjourney, etc.)
- Advanced editing UI with visual timeline
- Shot-to-video animation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with Google Gemini 2.5 Pro, Gemini 2.5 Flash Image, and Streamlit

Developed with assistance from Claude Code
