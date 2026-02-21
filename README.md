# Automated Research Digest Pipeline

A daily digest system that fetches academic papers from arXiv, extracts and
semantically chunks their text, stores embeddings in a vector database, and
delivers an LLM-generated summary via email.

## How It Works

```
arXiv API ──▶ PDF Download ──▶ Text Extraction (PyMuPDF)
                                       │
                                       ▼
                              Semantic Chunking (Chonkie)
                                       │
                                       ▼
                              Vector Storage (ChromaDB)
                                       │
                                       ▼
               GitHub Trending ──▶ LLM Summarization (OpenAI)
               (optional)              │
                                       ▼
                              Post-Processing (Implications & Critiques)
                                       │
                                       ▼
                              Email Dispatch (SMTP/SSL)
```

1. **Fetch** — Queries the arXiv API for papers in configured topics, filtered
   to the preceding 24-hour window. PDFs are downloaded with retry logic.
2. **Extract** — Uses PyMuPDF to pull raw text from each PDF. Image-only
   documents are flagged as unparseable.
3. **Chunk** — Splits extracted text into semantic segments using Chonkie's
   `SemanticChunker` with the `all-MiniLM-L6-v2` embedding model.
4. **Store** — Persists chunks with embeddings and metadata (title, authors,
   URL, date) in ChromaDB.
5. **GitHub Trending** *(optional)* — Fetches recently-created trending
   repositories from GitHub and appends them to the LLM prompt.
6. **Summarize** — Sends paper abstracts (and optional GitHub section) to an
   OpenAI-compatible LLM for digest generation.
7. **Post-process** — Optionally generates practical implications and critical
   analysis via separate LLM calls.
8. **Email** — Delivers the digest as a styled HTML + plaintext email via
   SMTP/SSL, or prints to console in dry-run mode.

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key
- SMTP credentials (for email delivery)

### Installation

```bash
pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable | Description | Default |
|---|---|---|
| `ARXIV_TOPICS` | Comma-separated arXiv categories | `cs.AI,cs.LG` |
| `ARXIV_MAX_RESULTS` | Max papers to fetch per run | `50` |
| `LLM_API_KEY` | OpenAI API key | — |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LLM_MAX_TOKENS` | Max tokens for LLM responses | `4096` |
| `LLM_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | `./data/chromadb` |
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port (SSL) | `465` |
| `SMTP_USER` | SMTP username | — |
| `SMTP_PASSWORD` | SMTP password / app password | — |
| `EMAIL_FROM` | Sender address | — |
| `EMAIL_TO` | Recipient address | — |
| `DRY_RUN` | Print to console instead of emailing | `true` |
| `GITHUB_ENABLED` | Enable GitHub trending section | `false` |
| `GITHUB_LANGUAGES` | Comma-separated languages to track | `python` |
| `GITHUB_TOP_N` | Number of trending repos to include | `5` |
| `POSTPROCESSING_IMPLICATIONS` | Enable practical implications section | `true` |
| `POSTPROCESSING_CRITIQUES` | Enable critical analysis section | `true` |

### Usage

```bash
# Dry-run (prints to console)
digest-pipeline --dry-run

# With specific topics
digest-pipeline --dry-run --topics cs.CL cs.CV

# Verbose logging
digest-pipeline --dry-run -v

# Production (sends email)
# Set DRY_RUN=false in .env, then:
digest-pipeline
```

## Project Structure

```
src/digest_pipeline/
├── __init__.py          # Package version
├── arxiv_topics.py      # Full arXiv taxonomy index and search utilities
├── chunker.py           # Semantic text chunking via Chonkie
├── config.py            # Centralized settings via pydantic-settings
├── emailer.py           # HTML/plaintext email formatting and SMTP dispatch
├── extractor.py         # PDF text extraction via PyMuPDF
├── fetcher.py           # arXiv paper fetching with retry-based PDF download
├── github_trending.py   # Optional GitHub trending repository module
├── pipeline.py          # Main orchestrator and CLI entry point
├── postprocessor.py     # LLM post-processing (implications & critiques)
├── summarizer.py        # LLM-powered summarization with backoff
└── vectorstore.py       # ChromaDB vector store for chunk storage

tests/
├── test_arxiv_topics.py
├── test_config.py
├── test_emailer.py
├── test_extractor.py
├── test_fetcher.py
├── test_github_trending.py
├── test_pipeline.py
├── test_postprocessor.py
└── test_summarizer.py

docs/
├── ears-design-document.md       # EARS requirements specification
└── e2e-integration-test-plan.md  # Integration & E2E testing plan
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Design Documentation

The system is specified using the EARS (Easy Approach to Requirements Syntax)
methodology. See [`docs/ears-design-document.md`](docs/ears-design-document.md)
for the full requirements specification including data schema definitions.

## License

Apache-2.0
