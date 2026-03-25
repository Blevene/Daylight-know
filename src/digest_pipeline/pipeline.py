"""Main pipeline orchestrator.

Ties together all modules to execute the full daily digest workflow:

  1. Fetch papers from arXiv RSS feed, HuggingFace, OpenAlex
  2. Deduplicate across sources and filter previously-seen papers
  3. Extract text from PDFs (pypdf) — or use abstract for PDF-less papers
  4. Chunk text semantically (Chonkie) and store in ChromaDB (ALL papers, parallel)
  5. Rank stored papers by interest relevance → select top N for digest
  6. (Optional) Fetch GitHub trending repos
  7. Summarize via LLM (per-paper JSON)
  8. Post-process: ELI5, implications & critiques (per-paper JSON, parallel)
  9. Assemble per-paper analyses and send email digest
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from digest_pipeline.chunker import chunk_text
from digest_pipeline.config import Settings, get_settings
from digest_pipeline.emailer import send_digest
from digest_pipeline.extractor import extract_text
from digest_pipeline.fetcher import Paper, download_pdf, fetch_papers
from digest_pipeline.github_trending import fetch_trending, format_for_prompt
from digest_pipeline.hf_fetcher import (
    HFDailyPaper,
    fetch_hf_daily,
    normalize_arxiv_id,
    reconcile_hf_papers,
)
from digest_pipeline.postprocessor import extract_implications, generate_critiques, generate_eli5
from digest_pipeline.openalex_fetcher import fetch_openalex_papers
from digest_pipeline.archiver import archive_papers
from digest_pipeline.ranker import rank_papers
from digest_pipeline.seen_papers import filter_unseen, load_seen, record_papers, save_seen
from digest_pipeline.summarizer import summarize
from digest_pipeline.vectorstore import VectorStoreError, _get_collection, store_chunks, store_unparseable

logger = logging.getLogger(__name__)


@dataclass
class PaperAnalysis:
    """Per-paper grouped analysis for the digest email."""

    title: str
    url: str
    source: str = "arxiv"
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    upvotes: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    summary: str = ""
    eli5: str = ""
    implications: str = ""
    critique: str = ""


def _build_analyses(
    papers: list[Paper],
    summaries: dict[str, str],
    implications: dict[str, str],
    critiques: dict[str, str],
    *,
    eli5: dict[str, str] | None = None,
) -> list[PaperAnalysis]:
    """Zip LLM results into per-paper PaperAnalysis objects."""
    if eli5 is None:
        eli5 = {}
    analyses: list[PaperAnalysis] = []
    for i, paper in enumerate(papers, 1):
        key = f"paper_{i}"
        if key not in summaries:
            logger.warning("Missing summary for %s (%s).", key, paper.title)
        if implications and key not in implications:
            logger.warning("Missing implications for %s (%s).", key, paper.title)
        if critiques and key not in critiques:
            logger.warning("Missing critique for %s (%s).", key, paper.title)
        if eli5 and key not in eli5:
            logger.warning("Missing ELI5 for %s (%s).", key, paper.title)
        analyses.append(
            PaperAnalysis(
                title=paper.title,
                url=paper.url,
                source=paper.source,
                authors=paper.authors,
                categories=paper.categories,
                upvotes=paper.upvotes,
                fields_of_study=paper.fields_of_study,
                summary=summaries.get(key, ""),
                eli5=eli5.get(key, ""),
                implications=implications.get(key, ""),
                critique=critiques.get(key, ""),
            )
        )
    return analyses


def _cleanup_pdf_dirs(papers: list[Paper]) -> None:
    """Remove temporary PDF directories created during fetching."""
    cleaned: set[Path] = set()
    for paper in papers:
        if paper.pdf_path is not None:
            parent = paper.pdf_path.parent
            if parent not in cleaned and parent.name.startswith(("arxiv_pdfs_", "hf_pdfs_")):
                shutil.rmtree(parent, ignore_errors=True)
                cleaned.add(parent)
    if cleaned:
        logger.info("Cleaned up %d temporary PDF directory(s).", len(cleaned))


def _process_paper(
    paper: Paper,
    collection: chromadb.Collection,
    settings: Settings,
) -> Paper | None:
    """Process a single paper: extract text, chunk, store. Thread-safe.

    Returns the Paper on success, None if unparseable/skipped/error.
    """
    if paper.pdf_path is not None:
        extraction = extract_text(paper.pdf_path, paper.paper_id)

        if not extraction.parseable:
            try:
                store_unparseable(paper, settings, collection=collection)
            except VectorStoreError:
                logger.error("ChromaDB store failed for unparseable paper %s.", paper.paper_id)
                return None
            return None

        text = extraction.text
    else:
        text = paper.abstract
        if not text:
            logger.warning("Paper %s has no PDF and no abstract — skipping.", paper.paper_id)
            return None

    chunks = chunk_text(text)

    try:
        store_chunks(paper, chunks, settings, collection=collection)
    except VectorStoreError:
        logger.error("ChromaDB store failed for paper %s.", paper.paper_id)
        return None

    return paper


def _ingest_papers(papers: list[Paper], settings: Settings) -> list[Paper]:
    """Extract, chunk, and store all papers. Parallel when workers > 1."""
    # Warm chunker cache before pool starts (avoids race on first call).
    chunk_text("warmup")

    # Create shared ChromaDB collection handle for all workers.
    try:
        collection = _get_collection(settings)
    except VectorStoreError:
        logger.critical("ChromaDB connection failed — halting pipeline.")
        sys.exit(1)

    workers = settings.pipeline_ingest_workers

    if workers <= 1:
        # Sequential fallback
        return [p for paper in papers if (p := _process_paper(paper, collection, settings)) is not None]

    processed: list[Paper] = []
    consecutive_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_paper = {
            executor.submit(_process_paper, paper, collection, settings): paper
            for paper in papers
        }

        for future in concurrent.futures.as_completed(future_to_paper):
            try:
                result = future.result()
            except Exception:
                paper = future_to_paper[future]
                logger.exception("Unexpected error processing paper %s.", paper.paper_id)
                result = None

            if result is not None:
                processed.append(result)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    logger.critical(
                        "5 consecutive paper failures — likely systemic. Cancelling remaining."
                    )
                    for f in future_to_paper:
                        f.cancel()
                    break

    return processed


def _postprocess(
    papers: list[Paper],
    settings: Settings,
    github_section: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Run summarization then post-processing (parallel or sequential).

    Returns (summaries, implications, critiques, eli5).
    """
    summaries = summarize(papers, settings, github_section=github_section)

    implications: dict[str, str] = {}
    critiques: dict[str, str] = {}
    eli5_results: dict[str, str] = {}

    # Build list of enabled post-processors.
    tasks: list[tuple[str, object]] = []
    if settings.postprocessing_implications:
        tasks.append(("implications", extract_implications))
    if settings.postprocessing_critiques:
        tasks.append(("critiques", generate_critiques))
    if settings.postprocessing_eli5:
        tasks.append(("eli5", generate_eli5))

    if not tasks:
        return summaries, implications, critiques, eli5_results

    if settings.pipeline_postprocess_parallel and len(tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {
                executor.submit(fn, papers, settings): name
                for name, fn in tasks
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Post-processing '%s' failed.", name)
                    result = {}
                if name == "implications":
                    implications = result
                elif name == "critiques":
                    critiques = result
                elif name == "eli5":
                    eli5_results = result
    else:
        # Sequential fallback
        for name, fn in tasks:
            try:
                result = fn(papers, settings)
            except Exception:
                logger.exception("Post-processing '%s' failed.", name)
                result = {}
            if name == "implications":
                implications = result
            elif name == "critiques":
                critiques = result
            elif name == "eli5":
                eli5_results = result

    # Error logging for empty results (preserve existing asymmetry).
    if settings.postprocessing_critiques and not critiques:
        logger.error("Critique generation returned no results for %d papers.", len(papers))
    if settings.postprocessing_eli5 and not eli5_results:
        logger.error("ELI5 generation returned no results for %d papers.", len(papers))

    return summaries, implications, critiques, eli5_results


def run(settings: Settings | None = None) -> None:
    """Execute the full pipeline once."""
    if settings is None:
        settings = get_settings()

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Pipeline run started for %s (dry_run=%s).", date_str, settings.dry_run)

    # ── Step 1: Fetch papers from all enabled sources ───────────
    # Always fetch the full pool; ranking happens after storage.
    arxiv_fetch_size = settings.arxiv_fetch_pool if (
        settings.interest_profile or settings.interest_keywords
    ) else settings.arxiv_max_results

    papers = fetch_papers(settings, max_results=arxiv_fetch_size)

    # HuggingFace: deduplicate against arXiv, split into new vs trending
    hf_trending: list[HFDailyPaper] = []
    if settings.huggingface_enabled:
        hf_daily = fetch_hf_daily(settings)
        # Normalize all arXiv IDs (strips version suffixes like "v2") so that
        # e.g. "2401.00001v2" from arXiv matches "2401.00001" from HuggingFace.
        known_ids = {normalize_arxiv_id(p.paper_id) for p in papers}
        hf_new, hf_trending = reconcile_hf_papers(hf_daily, known_ids)
        papers.extend(hf_new)

        # Download PDFs for HuggingFace-only papers via their arXiv ID.
        for paper in hf_new:
            # paper_id is "hf_{arxiv_id}" — extract the raw arXiv ID.
            raw_arxiv_id = paper.paper_id.removeprefix("hf_")
            pdf_url = f"https://arxiv.org/pdf/{raw_arxiv_id}"
            pdf_dest = Path(tempfile.mkdtemp(prefix="hf_pdfs_")) / f"{raw_arxiv_id}.pdf"
            if download_pdf(pdf_url, pdf_dest):
                paper.pdf_path = pdf_dest
            else:
                logger.info("No PDF for HF paper %s — will use abstract.", paper.paper_id)

    if settings.openalex_enabled:
        # Build set of known DOIs to avoid duplicating arXiv/HF papers.
        all_known_dois: set[str] = set()
        for p in papers:
            if p.source == "arxiv":
                raw_id = p.paper_id.split("v")[0]  # strip version
                all_known_dois.add(f"10.48550/arXiv.{raw_id}")
            elif p.source == "huggingface":
                raw_id = p.paper_id.removeprefix("hf_").split("v")[0]
                all_known_dois.add(f"10.48550/arXiv.{raw_id}")
        oa_papers = fetch_openalex_papers(settings, known_paper_ids=all_known_dois)
        papers.extend(oa_papers)

    # ── Cross-day deduplication ──────────────────────────────────
    seen = load_seen()
    papers = filter_unseen(papers, seen)

    if not papers:
        logger.warning("No papers found from any source. Exiting.")
        return

    # ── Step 2-4: Extract → Chunk → Store (all papers, parallel) ──
    processed_papers = _ingest_papers(papers, settings)

    # ── Archive PDFs (optional) ──────────────────────────────────
    if settings.pdf_archive_dir:
        archive_papers(papers, date_str, settings)

    # Clean up downloaded PDFs — text has been extracted and stored.
    _cleanup_pdf_dirs(papers)

    if not processed_papers:
        logger.warning("All papers were unparseable. Nothing to summarize.")
        return

    logger.info("Stored %d papers in vector store.", len(processed_papers))

    # ── Step 5: Rank for digest selection ─────────────────────────
    # Separate papers by source so each gets its own ranking pool/limit.
    arxiv_papers = [p for p in processed_papers if p.source == "arxiv"]
    hf_papers = [p for p in processed_papers if p.source == "huggingface"]
    oa_papers_stored = [p for p in processed_papers if p.source == "openalex"]

    digest_papers: list[Paper] = []

    if arxiv_papers:
        ranked_arxiv = rank_papers(arxiv_papers, settings, max_results=settings.arxiv_max_results)
        logger.info("arXiv: %d stored, %d selected for digest.", len(arxiv_papers), len(ranked_arxiv))
        digest_papers.extend(ranked_arxiv)

    digest_papers.extend(hf_papers)

    if oa_papers_stored:
        ranked_oa = rank_papers(
            oa_papers_stored,
            settings,
            interest_profile=settings.openalex_interest_profile or settings.interest_profile,
            interest_keywords=settings.openalex_interest_keywords or settings.interest_keywords,
            max_results=settings.openalex_max_results,
        )
        logger.info("OpenAlex: %d stored, %d selected for digest.", len(oa_papers_stored), len(ranked_oa))
        digest_papers.extend(ranked_oa)

    if not digest_papers:
        logger.warning("No papers selected for digest after ranking.")
        return

    processed_papers = digest_papers

    # ── Step 6: Optional GitHub trending ────────────────────────
    github_section = ""
    if settings.github_enabled:
        trending = fetch_trending(settings)
        github_section = format_for_prompt(trending)

    # ── Step 7: Post-processing (parallel or sequential) ─────────
    summaries, implications, critiques, eli5_results = _postprocess(
        processed_papers, settings, github_section
    )

    # ── Step 8: Assemble & send ─────────────────────────────────
    analyses = _build_analyses(processed_papers, summaries, implications, critiques, eli5=eli5_results)

    send_digest(
        analyses,
        date_str,
        settings,
        hf_trending=hf_trending,
    )

    # ── Record digested papers for cross-day dedup ────────────
    record_papers(processed_papers, seen, date_str)
    save_seen(seen, max_age_days=settings.dedup_history_days)

    logger.info("Pipeline run complete. %d paper(s) in digest.", len(processed_papers))


def _add_run_args(parser: argparse.ArgumentParser, *, is_subparser: bool = False) -> None:
    """Add run-mode arguments to a parser.

    When *is_subparser* is True, defaults use ``SUPPRESS`` so the
    subparser doesn't clobber values set by the root parser
    (e.g. ``digest-pipeline --dry-run run``).
    """
    default = argparse.SUPPRESS if is_subparser else None
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=default,
        help="Print email to console instead of sending (overrides .env)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=default,
        help="arXiv category codes (e.g. cs.AI cs.LG)",
    )


def _handle_run(args: argparse.Namespace) -> None:
    """Execute the pipeline from CLI args."""
    settings = get_settings()
    if args.dry_run:
        settings.dry_run = True
    if args.topics:
        settings.arxiv_topics = args.topics
    run(settings)


def main() -> None:
    """CLI entry point with subcommands: run, setup, topics."""
    parser = argparse.ArgumentParser(description="Automated Research Digest Pipeline")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    # Root-level run args for backward compat (digest-pipeline --dry-run)
    _add_run_args(parser)

    subparsers = parser.add_subparsers(dest="command")

    # ── run subcommand ───────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the digest pipeline")
    _add_run_args(run_parser, is_subparser=True)

    # ── setup subcommand ─────────────────────────────────────────
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )

    # ── topics subcommand ────────────────────────────────────────
    topics_parser = subparsers.add_parser("topics", help="Browse arXiv topics")
    topics_sub = topics_parser.add_subparsers(dest="topics_command")

    topics_sub.add_parser("list", help="List all topic groups")

    search_parser = topics_sub.add_parser("search", help="Search topics by keyword")
    search_parser.add_argument("query", help="Search query")

    group_parser = topics_sub.add_parser("group", help="List topics in a group")
    group_parser.add_argument("name", help="Group name (e.g. cs, math, physics)")

    validate_parser = topics_sub.add_parser("validate", help="Validate topic codes")
    validate_parser.add_argument("codes", nargs="+", help="Topic codes to validate")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "setup":
        from digest_pipeline.setup import run_setup_wizard

        run_setup_wizard(args.env_file)
    elif args.command == "topics":
        from digest_pipeline.topics_cli import handle_topics_command

        handle_topics_command(args)
    else:
        # No subcommand or "run" — execute pipeline
        _handle_run(args)


if __name__ == "__main__":
    main()
