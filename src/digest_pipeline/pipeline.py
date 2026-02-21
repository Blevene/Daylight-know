"""Main pipeline orchestrator.

Ties together all modules to execute the full daily digest workflow:

  1. Fetch papers from arXiv (24-hour window)
  2. Extract text from PDFs (PyMuPDF)
  3. Chunk text semantically (Chonkie)
  4. Store chunks + embeddings in ChromaDB
  5. (Optional) Fetch GitHub trending repos
  6. Summarize via LLM
  7. Send email digest (or print in dry-run mode)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from digest_pipeline.chunker import chunk_text
from digest_pipeline.config import Settings, get_settings
from digest_pipeline.emailer import send_digest
from digest_pipeline.extractor import extract_text
from digest_pipeline.fetcher import fetch_papers
from digest_pipeline.github_trending import fetch_trending, format_for_prompt
from digest_pipeline.postprocessor import extract_implications, generate_critiques
from digest_pipeline.summarizer import summarize
from digest_pipeline.vectorstore import VectorStoreError, store_chunks, store_unparseable

logger = logging.getLogger(__name__)


def run(settings: Settings | None = None) -> None:
    """Execute the full pipeline once."""
    if settings is None:
        settings = get_settings()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Pipeline run started for %s (dry_run=%s).", date_str, settings.dry_run)

    # ── Step 1: Fetch papers ────────────────────────────────────
    papers = fetch_papers(settings)
    if not papers:
        logger.warning("No papers found in the 24-hour window. Exiting.")
        return

    # ── Step 2-4: Extract → Chunk → Store ───────────────────────
    processed_papers = []
    for paper in papers:
        extraction = extract_text(paper.pdf_path, paper.arxiv_id)

        if not extraction.parseable:
            try:
                store_unparseable(paper, settings)
            except VectorStoreError:
                logger.critical("ChromaDB connection failed — halting pipeline.")
                sys.exit(1)
            continue

        chunks = chunk_text(extraction.text)

        try:
            store_chunks(paper, chunks, settings)
        except VectorStoreError:
            logger.critical("ChromaDB connection failed — halting pipeline.")
            sys.exit(1)

        processed_papers.append(paper)

    if not processed_papers:
        logger.warning("All papers were unparseable. Nothing to summarize.")
        return

    # ── Step 5: Optional GitHub trending ────────────────────────
    github_section = ""
    if settings.github_enabled:
        trending = fetch_trending(settings)
        github_section = format_for_prompt(trending)

    # ── Step 6: LLM summarization ───────────────────────────────
    summary = summarize(processed_papers, settings, github_section=github_section)

    # ── Step 6b: Post-processing (implications & critiques) ───
    implications = ""
    if settings.postprocessing_implications:
        implications = extract_implications(processed_papers, settings)

    critiques = ""
    if settings.postprocessing_critiques:
        critiques = generate_critiques(processed_papers, settings)

    # ── Step 7: Email dispatch ──────────────────────────────────
    send_digest(
        summary,
        len(processed_papers),
        date_str,
        settings,
        implications=implications,
        critiques=critiques,
    )

    logger.info("Pipeline run complete. %d paper(s) in digest.", len(processed_papers))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Automated Research Digest Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Print email to console instead of sending (overrides .env)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="arXiv category codes (e.g. cs.AI cs.LG)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    if args.dry_run is not None:
        settings.dry_run = True
    if args.topics:
        settings.arxiv_topics = args.topics

    run(settings)


if __name__ == "__main__":
    main()
