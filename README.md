# Story Architect

**AI-Powered Multi-Agent Pipeline for Script-to-Shot Conversion**

Story Architect is a production-ready system that transforms story concepts and screenplays into detailed shot breakdowns optimized for AI video generation. Powered by Google Gemini 2.5 Pro, it uses a specialized four-agent pipeline to generate comprehensive visual prompts ready for text-to-video models.

## Features

- **Multi-Agent Pipeline**: Four specialized AI agents working sequentially to transform stories into shots
- **Flexible Entry Points**: Start from a logline or bring your own screenplay
- **Session Management**: Save progress and resume from any agent in the pipeline
- **Smart Error Handling**: Automatic retry with feedback loop (up to 3 attempts) plus manual intervention
- **Modular Design**: Each agent can be customized independently via prompt templates
- **Interactive GUI**: Clean Streamlit interface for project management
- **Export Capabilities**: Download all outputs in JSON/text formats

## Architecture

### Four-Agent Pipeline

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
  - First Frame: Verbose visual description for image generation (includes all elements)
  - Animation: Instructions for animating the first frame for video generation
- Format: Strict JSON schema with validation

**Agent 4: Shot Grouping**
- Input: Shot breakdown
- Output: Parent/child shot relationships in nested hierarchy
- Purpose: Optimize generation by grouping similar shots for edit-based models
- Features: Cross-scene grouping, multi-level nesting
- Format: Nested JSON

### Tech Stack

- Python 3.10+
- Google Gemini 2.5 Pro (via Gemini API or Vertex AI)
- Streamlit (GUI)
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
git clone https://github.com/yourusername/story-architect.git
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
2. Choose starting point (Agent 1 for logline, Agent 2 for screenplay)
3. Enter your input text
4. Optional: name your project
5. Click "Run Pipeline"
6. Monitor progress across agent tabs

**Resume Session:**
1. Select "Resume Session" from sidebar
2. Choose previous session
3. Review existing outputs
4. Select agent to resume from
5. Click "Resume Pipeline"

**Session History:**
- View all past sessions
- Browse outputs
- Delete old sessions

### CLI Usage

Run pipeline:
```bash
python main.py run input.txt --start-agent agent_1 --name "My Project"
```

Resume session:
```bash
python main.py resume <session_id> --from-agent agent_2
```

List sessions:
```bash
python main.py list --limit 20
```

### Output Structure

All outputs saved to `outputs/projects/<session_id>/`:
- `session_state.json` - Complete session state
- `agent_1_output.txt` - Generated screenplay
- `agent_2_output.json` - Scene breakdown
- `agent_3_output.json` - Shot breakdown with visual prompts
- `agent_4_output.json` - Grouped shot hierarchy

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
```

### Prompt Templates

All prompts are in `prompts/` as `.txt` files:
- Use `{input}` placeholder for input data
- Changes take effect immediately
- No code modification needed

## Project Structure

```
story-architect/
├── agents/                    # Agent implementations
│   ├── base_agent.py         # Abstract base class
│   ├── agent_1_screenplay.py
│   ├── agent_2_scene_breakdown.py
│   ├── agent_3_shot_breakdown.py
│   └── agent_4_grouping.py
├── core/                      # Core components
│   ├── gemini_client.py      # API wrapper with retry logic
│   ├── pipeline.py           # Pipeline orchestration
│   ├── session_manager.py    # Session persistence
│   ├── validators.py         # Pydantic schemas
│   └── export.py             # Export utilities
├── gui/                       # Streamlit GUI
│   └── app.py
├── prompts/                   # Agent prompt templates
│   ├── agent_1_prompt.txt
│   ├── agent_2_prompt.txt
│   ├── agent_3_prompt.txt
│   └── agent_4_prompt.txt
├── examples/                  # Example input files
├── config.yaml               # Configuration
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
└── main.py                   # CLI entry point
```

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

**Agent Validation Failures**
- System auto-retries up to 3 times with feedback
- Check error messages in GUI
- Review intermediate outputs in session history
- Manually edit outputs and resume if needed

## Billing & Rate Limits

The Gemini API tier (free vs paid) is determined by your Google AI Studio billing setup, not by code configuration.

**Free Tier:**
- 15 requests per minute
- Limited features
- Data may be used to improve Google products

**Paid Tier:**
- 1000+ requests per minute (model-dependent)
- Full API features (context caching, batch API)
- Data NOT used for model improvement
- Enable at: https://aistudio.google.com/api-keys

See pricing: https://ai.google.dev/gemini-api/docs/pricing

## Known Limitations

- No video generation integration (outputs prompts only)
- Single model support (Gemini 2.5 Pro)
- Manual JSON editing required for session modifications
- No batch processing

## Future Roadmap

- Video generation integration
- Visual shot preview
- Character/location database
- Batch processing
- Multi-model support
- Advanced editing UI

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with Google Gemini 2.5 Pro and Streamlit
