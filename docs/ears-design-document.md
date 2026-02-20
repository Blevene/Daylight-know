# Design Document: Automated Research Digest Pipeline

## 1. System Overview

The Automated Research Digest Pipeline is an ingestion, storage, and summarization system. It automatically fetches daily academic papers from arXiv, extracts and semantically chunks the text using Chonkie, stores the embeddings in a Vector Database (ChromaDB), and delivers an LLM-generated summary via email. It is designed to be extensible for future data sources, such as GitHub repositories.

## 2. EARS Requirements Specification

### 2.1 Ubiquitous Requirements (Always Active)

These requirements define fundamental, constant behaviors of the system.

- **The** Pipeline **shall** use PyMuPDF to extract raw text from downloaded PDF documents.
- **The** Pipeline **shall** utilize the Chonkie `SemanticChunker` to segment extracted text based on semantic boundaries rather than fixed character counts.
- **The** Pipeline **shall** store all generated text chunks, along with their associated embeddings and metadata (Title, Authors, URL, Date), in a ChromaDB Vector Database.
- **The** Pipeline **shall** format the final daily digest as an HTML/Plaintext email.

### 2.2 Event-Driven Requirements (Triggered Behaviors)

These requirements define how the system reacts to specific triggers (events).

- **When** the daily scheduled cron job triggers, **the** Pipeline **shall** query the arXiv API for papers within the configured topic(s).
- **When** the arXiv API returns the search results, **the** Pipeline **shall** filter the results to only include papers submitted within the preceding 24-hour window.
- **When** all PDF texts have been chunked and stored in the Vector Database, **the** Pipeline **shall** pass the paper abstracts to the LLM for summarization.
- **When** the LLM successfully generates the summary text, **the** Pipeline **shall** dispatch the email to the configured recipient address using SMTP.

### 2.3 State-Driven Requirements (Behaviors in a Specific State)

These requirements apply only while the system is in a defined state.

- **While** querying the LLM API for summarization, **the** Pipeline **shall** enforce a strict maximum token limit to prevent context window overflow.
- **While** connecting to the SMTP server, **the** Pipeline **shall** use an encrypted SSL/TLS connection.
- **While** running in "dry-run" or "testing" mode, **the** Pipeline **shall** output the final email payload to the console instead of sending the email.

### 2.4 Unwanted Behavior Requirements (Error Handling)

These requirements define how the system recovers from or handles failures.

- **If** a specific arXiv PDF fails to download after 3 attempts, **then** the Pipeline **shall** log the failure, skip the paper, and proceed to the next item in the queue.
- **If** the text extraction results in an empty string (e.g., scanned images without OCR), **then** the Pipeline **shall** flag the document metadata as "unparseable" in the database and skip chunking.
- **If** the ChromaDB connection fails, **then** the Pipeline **shall** halt execution and trigger an alert.
- **If** the LLM API rate limit is exceeded, **then** the Pipeline **shall** implement an exponential backoff strategy before retrying the request.

### 2.5 Optional Feature Requirements (Future Extensions)

These requirements cover features that may be toggled on or added later.

- **Where** the GitHub Trending module is enabled, **the** Pipeline **shall** query the GitHub Search API for repositories created in the last 24 hours matching the user's configured programming languages.
- **Where** the GitHub Trending module is enabled, **the** Pipeline **shall** append the top 5 repository names, descriptions, and URLs to the LLM summarization prompt before the final email is generated.

## 3. Data Schema Definitions

To support the above requirements, the ChromaDB metadata schema must be strictly defined to ensure smooth semantic retrieval later.

### ChromaDB Metadata Mapping

| Field            | Type      | Description                                          |
|------------------|-----------|------------------------------------------------------|
| `doc_id`         | String    | Unique identifier (e.g., `<arxiv_id>_chunk_<index>`) |
| `source`         | String    | Data origin (e.g., `"arxiv"` or `"github"`)          |
| `title`          | String    | Paper or repository title                            |
| `authors`        | String    | Comma-separated list of authors                      |
| `url`            | String    | Source URL                                           |
| `published_date` | String    | ISO 8601 Timestamp                                   |
| `chunk_index`    | Integer   | Order of the chunk in the original text              |
