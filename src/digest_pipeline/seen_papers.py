"""Cross-day paper deduplication via a persistent JSON ledger.

Tracks paper IDs that have already appeared in a digest so they are not
repeated on subsequent runs.  Entries older than ``max_age_days`` are
automatically pruned on each save to prevent unbounded growth.

The ledger lives at ``data/seen_papers.json`` by default.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/seen_papers.json")


def load_seen(path: Path = DEFAULT_LEDGER_PATH) -> dict[str, str]:
    """Load the seen-papers ledger from *path*.

    Returns a dict mapping ``paper_id`` → ISO date string.
    If the file does not exist or is unreadable, returns an empty dict.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        logger.warning("Seen-papers ledger is not a dict — resetting.")
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read seen-papers ledger: %s", exc)
        return {}


def save_seen(
    seen: dict[str, str],
    path: Path = DEFAULT_LEDGER_PATH,
    *,
    max_age_days: int = 30,
) -> None:
    """Write the seen-papers ledger to *path*, pruning old entries.

    Entries whose date is more than *max_age_days* ago are dropped.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    pruned = {pid: date for pid, date in seen.items() if date >= cutoff}

    if len(pruned) < len(seen):
        logger.info("Pruned %d stale entries from seen-papers ledger.", len(seen) - len(pruned))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pruned, indent=2) + "\n", encoding="utf-8")


def filter_unseen(
    papers: list[Paper],
    seen: dict[str, str],
) -> list[Paper]:
    """Return only papers whose ``paper_id`` is not already in *seen*."""
    unseen = [p for p in papers if p.paper_id not in seen]
    skipped = len(papers) - len(unseen)
    if skipped:
        logger.info("Skipped %d previously-seen paper(s).", skipped)
    return unseen


def record_papers(
    papers: list[Paper],
    seen: dict[str, str],
    date_str: str,
) -> dict[str, str]:
    """Add *papers* to the *seen* dict under *date_str* and return it."""
    for p in papers:
        seen[p.paper_id] = date_str
    return seen
