# Automated Research Digest Pipeline

A daily digest system that fetches academic papers from arXiv, HuggingFace
Daily Papers, and OpenAlex, then extracts and semantically chunks their text,
stores embeddings in a vector database, and delivers an LLM-generated summary
via email. Supports 150+ arXiv categories, 26 OpenAlex academic fields, and
any LLM provider through [litellm](https://github.com/BerriAI/litellm).

## How It Works

```
arXiv API ─────────────┐
                       ├──> PDF Download ──> Text Extraction (PyMuPDF)
HuggingFace Daily ─────┤                            |
  (optional)           │                            v
                       │                   Semantic Chunking (Chonkie)
OpenAlex API ──────────┤                            |
  (optional,           │                            v
   with ranking) ──────┘                   Vector Storage (ChromaDB)
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
   configurable retry logic. Optionally also fetches from HuggingFace
   Daily Papers (community-upvoted) and OpenAlex (broad academic coverage
   across 26 fields). Papers are deduplicated across sources by DOI.
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
- **Metadata** — Authors, categories/fields of study, source, and direct links

## Getting Started

### Prerequisites

- Python 3.10+
- An API key for any [litellm-supported LLM provider](https://docs.litellm.ai/docs/providers)
- SMTP credentials for email delivery (e.g., a Gmail App Password)

### Installation

```bash
git clone https://github.com/Blevene/Daylight-know.git
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
- `gemini/gemini-pro-latest` — Google (latest Pro model)
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

#### HuggingFace Daily Papers (Optional)

| Variable | Description | Default |
|---|---|---|
| `HUGGINGFACE_ENABLED` | Include HuggingFace community papers | `false` |
| `HUGGINGFACE_TOKEN` | HuggingFace API token (for rate limits) | — |
| `HUGGINGFACE_MAX_RESULTS` | Max HuggingFace papers | `20` |

Papers from HuggingFace are deduplicated against arXiv — those already
fetched from arXiv appear as "trending" in a sidebar rather than being
processed twice.

#### OpenAlex (Optional)

| Variable | Description | Default |
|---|---|---|
| `OPENALEX_ENABLED` | Include OpenAlex academic papers | `false` |
| `OPENALEX_API_KEY` | OpenAlex API key (required since Feb 2026) | — |
| `OPENALEX_EMAIL` | Email for polite pool (recommended) | — |
| `OPENALEX_MAX_RESULTS` | Papers to include in digest | `20` |
| `OPENALEX_QUERY` | Search query (used when ranking is off) | `machine learning` |
| `OPENALEX_FIELDS` | JSON array of academic fields to filter | — |

Valid fields: `Computer Science`, `Mathematics`, `Physics and Astronomy`,
`Chemistry`, `Engineering`, `Medicine`, `Psychology`, and 19 others
(26 total). See the setup wizard for the full list.

#### Interest-Based Ranking (Optional, requires OpenAlex)

When configured, the pipeline fetches a larger pool of OpenAlex papers and
uses LLM scoring to select only the most relevant ones for your digest.

| Variable | Description | Default |
|---|---|---|
| `OPENALEX_INTEREST_PROFILE` | Natural language description of your research interests | — |
| `OPENALEX_INTEREST_KEYWORDS` | JSON array of boost keywords | — |
| `OPENALEX_FETCH_POOL` | Papers to fetch before ranking | `100` |

**How it works:** The ranker scores each paper using two signals:
1. **Keyword boost** — +2 points per keyword match in title or abstract
2. **LLM scoring** — Papers are sent in batches of 20 to your configured
   LLM, which rates each paper 1-10 against your interest profile

Papers are ranked by combined score and the top `OPENALEX_MAX_RESULTS` are
kept. If the LLM call fails, keyword-only ranking is used as a fallback.
If neither profile nor keywords are configured, all fetched papers pass
through (backward compatible).

**Example:**

```env
OPENALEX_INTEREST_PROFILE="AI applications including world models, frontier AI methods, memory and retrieval systems, and cybersecurity"
OPENALEX_INTEREST_KEYWORDS=["world model","RAG","knowledge graph","cybersecurity","LLM","reasoning","agent"]
OPENALEX_FETCH_POOL="100"
```

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

The pipeline fetches papers from the last 24 hours, so running it once
daily on weekdays is ideal. arXiv publishes new submissions Sun-Thu
around 20:00 UTC and does not publish on weekends, so a Monday-Friday
schedule catches every batch.

**1. Find your executable path:**

```bash
which digest-pipeline
```

**2. Edit your crontab:**

```bash
crontab -e
```

**3. Add the pipeline and log rotation:**

```cron
# Research digest pipeline - Mon-Fri at 7:00 AM EST (12:00 UTC)
0 12 * * 1-5 cd /path/to/Daylight-know && /path/to/bin/digest-pipeline >> /path/to/Daylight-know/logs/digest-pipeline.log 2>&1

# Log rotation - keep previous week's log, truncate current (Monday midnight)
0 0 * * 1 cp /path/to/Daylight-know/logs/digest-pipeline.log /path/to/Daylight-know/logs/digest-pipeline.log.prev && : > /path/to/Daylight-know/logs/digest-pipeline.log
```

Replace `/path/to/Daylight-know` with your project directory and
`/path/to/bin/digest-pipeline` with the output of `which digest-pipeline`.

**Important:**

- **Working directory:** The `cd` is required so the pipeline finds your
  `.env` file and the relative ChromaDB storage path resolves correctly.
- **Logs directory:** Create `logs/` in the project root before the first
  run: `mkdir -p logs`
- **Timing:** 12:00 UTC = 7:00 AM EST. Adjust for your timezone. Morning
  runs give the best coverage since arXiv publishes the previous evening.
- **Log rotation:** The Monday midnight job copies the current log to
  `.log.prev` and truncates the current file, keeping two weeks of history.

**4. Verify it works:**

```bash
# Test the exact command cron will run
cd /path/to/Daylight-know && digest-pipeline
```

**5. Set production mode:**

Make sure `DRY_RUN=false` in your `.env` before the first scheduled run,
and verify your SMTP credentials work with a manual test run first.

### Scheduling with systemd (Linux)

For more robust scheduling with automatic logging via `journalctl` and
`Persistent=true` (runs missed jobs on next boot):

```ini
# /etc/systemd/system/digest-pipeline.service
[Unit]
Description=Research Digest Pipeline
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/Daylight-know
ExecStart=/path/to/bin/digest-pipeline
User=your-username
```

```ini
# /etc/systemd/system/digest-pipeline.timer
[Unit]
Description=Run Research Digest Pipeline weekdays

[Timer]
OnCalendar=Mon..Fri *-*-* 12:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now digest-pipeline.timer
sudo systemctl status digest-pipeline.timer
journalctl -u digest-pipeline.service  # view logs
```

## Tech Stack

### Core Libraries

| Library | Version | Role |
|---|---|---|
| [litellm](https://github.com/BerriAI/litellm) | >=1.30 | Universal LLM gateway — routes to OpenAI, Anthropic, Google, Ollama, Azure, and 100+ other providers |
| [ChromaDB](https://www.trychroma.com/) | >=0.5 | Local vector store with persistent SQLite + HNSW indexing for chunk storage and retrieval |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | >=1.24 | PDF text extraction |
| [Chonkie](https://github.com/chonkie-ai/chonkie) | >=1.0 | Semantic text chunking with the `potion-base-32M` embedding model |
| [arxiv](https://github.com/lukasschwab/arxiv.py) | >=2.1 | arXiv API client for paper fetching |
| [Pydantic](https://docs.pydantic.dev/) | >=2.0 | Data validation and schema definitions (Paper, TextChunk models) |
| [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | >=2.0 | Configuration management from environment variables and `.env` files |
| [Jinja2](https://jinja.palletsprojects.com/) | >=3.1 | HTML and plaintext email templating |
| [Mistune](https://mistune.lepture.com/) | >=3.0 | Markdown-to-HTML conversion for email rendering |
| [Rich](https://rich.readthedocs.io/) | >=13.0 | Interactive setup wizard TUI (console, tables, panels, prompts) |
| [Requests](https://requests.readthedocs.io/) | >=2.31 | HTTP client for HuggingFace, OpenAlex, and GitHub APIs |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | >=1.0 | `.env` file loading |

### Dev & Testing

| Library | Role |
|---|---|
| [pytest](https://pytest.org/) | Test runner with custom markers (`unit`, `integration`, `e2e`, `network`) |
| [Ruff](https://docs.astral.sh/ruff/) | Linter and formatter (target: Python 3.10, line length: 100) |
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Stub LLM server for integration tests |
| [aiosmtpd](https://aiosmtpd.readthedocs.io/) | Fake SMTP server for email integration tests |
| [reportlab](https://www.reportlab.com/) | Test PDF fixture generation |
| [Hatchling](https://hatch.pypa.io/) | Build backend |

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
├── hf_fetcher.py        # HuggingFace Daily Papers fetching & deduplication
├── llm_utils.py         # Shared LLM call utilities (backoff, structured output)
├── openalex_fetcher.py  # OpenAlex paper fetching with field filtering
├── pipeline.py          # Main orchestrator and CLI entry point
├── postprocessor.py     # LLM post-processing (implications & critiques)
├── prompts/             # LLM prompt templates (Markdown)
│   ├── summarizer.md    # Summarization prompt
│   ├── implications.md  # Practical implications prompt
│   ├── critiques.md     # Structured critique prompt
│   └── ranker.md        # Interest-based relevance scoring prompt
├── ranker.py            # Interest-based paper ranking (keyword + LLM scoring)
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
