# Automated Research Digest Pipeline

A daily digest system that fetches academic papers from arXiv, extracts and
semantically chunks their text, stores embeddings in a vector database, and
delivers an LLM-generated summary via email. Supports 150+ arXiv categories
and any LLM provider through [litellm](https://github.com/BerriAI/litellm).

## How It Works

```
arXiv API ──> PDF Download ──> Text Extraction (PyMuPDF)
                                       |
                                       v
                              Semantic Chunking (Chonkie)
                                       |
                                       v
                              Vector Storage (ChromaDB)
                                       |
                                       v
               GitHub Trending ──> LLM Summarization (litellm)
               (optional)              |
                                       v
                              Post-Processing (Implications & Critiques)
                                       |
                                       v
                              Email Dispatch (SMTP/STARTTLS)
```

### Pipeline Steps

1. **Fetch** — Queries the arXiv API for papers in your configured topics,
   filtered to the preceding 24-hour window. PDFs are downloaded with
   configurable retry logic.
2. **Extract** — Uses PyMuPDF to pull raw text from each PDF. Image-only
   documents are flagged as unparseable and stored separately.
3. **Chunk** — Splits extracted text into semantic segments using Chonkie's
   `SemanticChunker` with the `potion-base-32M` embedding model.
4. **Store** — Persists chunks with embeddings and metadata (title, authors,
   URL, date, chunk index) in ChromaDB for future retrieval.
5. **GitHub Trending** *(optional)* — Fetches recently-created trending
   repositories from GitHub and appends them to the LLM prompt.
6. **Summarize** — Sends paper text to an LLM via litellm, producing
   per-paper structured summaries. Supports OpenAI, Anthropic, Google,
   Cohere, Ollama, Azure, and 100+ other providers.
7. **Post-process** — Optionally generates practical implications (who
   benefits, how to apply) and structured critiques (strengths, weaknesses,
   open questions) via separate LLM calls.
8. **Email** — Delivers the digest as a styled HTML + plaintext email via
   SMTP with STARTTLS, or prints to console in dry-run mode.

### What You Get

Each digest email contains per-paper sections with:

- **Summary** — Key findings and contributions
- **Practical Implications** — Who benefits and how to apply the research
- **Critique** — Methodological strengths, weaknesses, and open questions
- **Metadata** — Authors, arXiv categories, and direct links

## Getting Started

### Prerequisites

- Python 3.10+
- An API key for any [litellm-supported LLM provider](https://docs.litellm.ai/docs/providers)
- SMTP credentials for email delivery (e.g., a Gmail App Password)

### Installation

```bash
git clone https://github.com/your-org/Daylight-know.git
cd Daylight-know
pip install -e ".[dev]"
```

### Option A: Interactive Setup Wizard (Recommended)

The setup wizard walks you through configuring everything interactively,
with topic browsing, connection testing, and `.env` file generation:

```bash
digest-pipeline setup
```

The wizard will:

1. Help you **browse and select arXiv topics** — search by keyword, browse
   by group (CS, Math, Physics, etc.), or type codes directly
2. Configure your **LLM provider** and optionally test the connection
3. Configure **SMTP email** settings and optionally test delivery
4. Set up **ChromaDB** storage location
5. Toggle **optional features** (GitHub trending, implications, critiques)
6. Write a complete `.env` file (with backup if one exists)

### Option B: Manual Configuration

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

### Browsing arXiv Topics

Not sure which topics to subscribe to? Use the topic browser:

```bash
# List all topic groups with counts
digest-pipeline topics list

# Search by keyword
digest-pipeline topics search "machine learning"
digest-pipeline topics search "quantum"
digest-pipeline topics search "natural language"

# List all topics in a group
digest-pipeline topics group cs
digest-pipeline topics group physics

# Validate topic codes
digest-pipeline topics validate cs.AI cs.LG stat.ML
```

### Configuration Reference

All settings are configured via environment variables in `.env`:

#### arXiv Settings

| Variable | Description | Default |
|---|---|---|
| `ARXIV_TOPICS` | Comma-separated arXiv category codes | `cs.AI,cs.LG` |
| `ARXIV_MAX_RESULTS` | Max papers to fetch per run | `50` |

#### LLM Settings

| Variable | Description | Default |
|---|---|---|
| `LLM_MODEL` | litellm model string | `openai/gpt-4o-mini` |
| `LLM_API_KEY` | API key for your LLM provider | *required* |
| `LLM_MAX_TOKENS` | Max tokens for LLM responses | `4096` |
| `LLM_API_BASE` | Custom API base URL (optional) | — |

The `LLM_MODEL` uses litellm's provider/model format. Examples:

- `openai/gpt-4o-mini` — OpenAI
- `anthropic/claude-sonnet-4-20250514` — Anthropic
- `gemini/gemini-2.0-flash` — Google
- `ollama/llama3` — Local Ollama
- `azure/gpt-4o` — Azure OpenAI

See [litellm providers](https://docs.litellm.ai/docs/providers) for the
full list.

#### Email / SMTP Settings

| Variable | Description | Default |
|---|---|---|
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port (STARTTLS) | `587` |
| `SMTP_USER` | SMTP username (usually your email) | *required* |
| `SMTP_PASSWORD` | SMTP password or app password | *required* |
| `EMAIL_FROM` | Sender email address | *required* |
| `EMAIL_TO` | Recipient email address | *required* |

**Gmail users:** You'll need an [App Password](https://support.google.com/accounts/answer/185833).
Go to Google Account > Security > 2-Step Verification > App Passwords,
generate one for "Mail", and use it as `SMTP_PASSWORD`.

#### ChromaDB Settings

| Variable | Description | Default |
|---|---|---|
| `CHROMA_PERSIST_DIR` | Directory for ChromaDB storage | `./data/chromadb` |
| `CHROMA_COLLECTION` | Collection name | `research_digest` |

#### Pipeline Modes

| Variable | Description | Default |
|---|---|---|
| `DRY_RUN` | Print digest to console instead of emailing | `true` |
| `PDF_DOWNLOAD_MAX_RETRIES` | PDF download retry attempts | `3` |

#### Post-Processing

| Variable | Description | Default |
|---|---|---|
| `POSTPROCESSING_IMPLICATIONS` | Generate practical implications | `true` |
| `POSTPROCESSING_CRITIQUES` | Generate structured critiques | `true` |

#### GitHub Trending (Optional)

| Variable | Description | Default |
|---|---|---|
| `GITHUB_ENABLED` | Include trending repos in digest | `false` |
| `GITHUB_LANGUAGES` | Comma-separated languages to track | `python` |
| `GITHUB_TOP_N` | Number of trending repos | `5` |

## Usage

### Running the Pipeline

```bash
# Dry-run (prints digest to console)
digest-pipeline --dry-run

# With specific topics (overrides .env)
digest-pipeline --dry-run --topics cs.CL cs.CV

# Verbose logging
digest-pipeline --dry-run -v

# Production mode (sends email)
# Set DRY_RUN=false in .env, then:
digest-pipeline
```

You can also use the explicit `run` subcommand:

```bash
digest-pipeline run --dry-run --topics cs.AI
```

### Scheduling with Cron

To receive daily digests automatically, set up a cron job. The pipeline
fetches papers from the last 24 hours, so running it once daily is ideal.

**1. Find your Python path:**

```bash
which digest-pipeline
# or, if installed in a virtualenv:
/path/to/venv/bin/digest-pipeline
```

**2. Edit your crontab:**

```bash
crontab -e
```

**3. Add a daily schedule:**

```cron
# Run digest at 7:00 AM UTC every day
0 7 * * * cd /path/to/Daylight-know && /path/to/venv/bin/digest-pipeline >> /var/log/digest-pipeline.log 2>&1

# Or, run at 6:00 AM US Eastern (with timezone-aware scheduling)
0 11 * * * cd /path/to/Daylight-know && /path/to/venv/bin/digest-pipeline >> /var/log/digest-pipeline.log 2>&1
```

**Tips for cron scheduling:**

- **Timing:** arXiv publishes new submissions around 20:00 UTC (Sun-Thu),
  so scheduling your digest for early morning gives the best coverage of
  the previous day's papers.
- **Working directory:** Use `cd` to ensure the `.env` file and ChromaDB
  storage directory are found correctly.
- **Logging:** Redirect output to a log file for debugging.
- **Weekends:** arXiv doesn't publish on weekends (Fri/Sat nights), so
  a Monday morning run may return fewer results. Consider running
  Monday-Friday only: `0 7 * * 1-5`.

**4. Verify it works:**

```bash
# Test the exact command cron will run
cd /path/to/Daylight-know && /path/to/venv/bin/digest-pipeline
```

### Scheduling with systemd (Linux)

For more robust scheduling with logging and restart capabilities:

```ini
# /etc/systemd/system/digest-pipeline.service
[Unit]
Description=Research Digest Pipeline
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/Daylight-know
ExecStart=/path/to/venv/bin/digest-pipeline
User=your-username
```

```ini
# /etc/systemd/system/digest-pipeline.timer
[Unit]
Description=Run Research Digest Pipeline daily

[Timer]
OnCalendar=*-*-* 07:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now digest-pipeline.timer
sudo systemctl status digest-pipeline.timer
journalctl -u digest-pipeline.service  # view logs
```

## Project Structure

```
src/digest_pipeline/
├── __init__.py          # Package version
├── arxiv_topics.py      # Full arXiv taxonomy (150+ topics) with search/validate
├── chunker.py           # Semantic text chunking via Chonkie
├── config.py            # Centralized settings via pydantic-settings
├── emailer.py           # HTML/plaintext email formatting and SMTP dispatch
├── extractor.py         # PDF text extraction via PyMuPDF
├── fetcher.py           # arXiv paper fetching with retry-based PDF download
├── github_trending.py   # Optional GitHub trending repository module
├── pipeline.py          # Main orchestrator and CLI entry point
├── postprocessor.py     # LLM post-processing (implications & critiques)
├── prompts/             # LLM prompt templates (Markdown)
│   ├── summarizer.md    # Summarization prompt
│   ├── implications.md  # Practical implications prompt
│   └── critiques.md     # Structured critique prompt
├── setup.py             # Interactive setup wizard
├── summarizer.py        # LLM-powered summarization with backoff
├── topics_cli.py        # Topic browser CLI
└── vectorstore.py       # ChromaDB vector store for chunk storage

tests/
├── unit/                # Fast tests, no external dependencies
├── integration/         # Tests with real local dependencies (ChromaDB, PyMuPDF)
├── e2e/                 # Full pipeline runs
└── fixtures/            # Test PDFs and data
```

## Running Tests

```bash
# Run all unit tests
pytest tests/unit/ -q

# Run integration tests (requires local dependencies)
pytest tests/integration/ -q

# Run end-to-end tests (requires LLM API key)
pytest tests/e2e/ -q

# Run everything
pytest

# Run with markers
pytest -m unit          # fast, no external deps
pytest -m integration   # real local deps
pytest -m e2e           # full pipeline
pytest -m network       # requires internet
```

## Design Documentation

The system is specified using the EARS (Easy Approach to Requirements Syntax)
methodology. See [`docs/ears-design-document.md`](docs/ears-design-document.md)
for the full requirements specification including data schema definitions.

## License

Apache-2.0
