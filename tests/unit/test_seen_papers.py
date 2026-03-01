"""Tests for the cross-day seen-papers deduplication module."""

import json
from datetime import datetime, timedelta, timezone

from digest_pipeline.fetcher import Paper
from digest_pipeline.seen_papers import (
    filter_unseen,
    load_seen,
    record_papers,
    save_seen,
)


def _make_paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        authors=["Alice"],
        abstract="Abstract.",
        url=f"https://example.com/{paper_id}",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        source="arxiv",
        pdf_path=None,
    )


class TestLoadSeen:
    def test_missing_file(self, tmp_path):
        assert load_seen(tmp_path / "nope.json") == {}

    def test_valid_file(self, tmp_path):
        ledger = tmp_path / "seen.json"
        ledger.write_text('{"paper_1": "2025-01-15"}')
        assert load_seen(ledger) == {"paper_1": "2025-01-15"}

    def test_invalid_json(self, tmp_path):
        ledger = tmp_path / "seen.json"
        ledger.write_text("not json")
        assert load_seen(ledger) == {}

    def test_non_dict_json(self, tmp_path):
        ledger = tmp_path / "seen.json"
        ledger.write_text("[1, 2, 3]")
        assert load_seen(ledger) == {}


class TestSaveSeen:
    def test_basic_save(self, tmp_path):
        ledger = tmp_path / "seen.json"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_seen({"p1": today}, ledger)
        data = json.loads(ledger.read_text())
        assert data == {"p1": today}

    def test_prunes_old_entries(self, tmp_path):
        ledger = tmp_path / "seen.json"
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_seen({"old": old_date, "new": today}, ledger, max_age_days=30)
        data = json.loads(ledger.read_text())
        assert "old" not in data
        assert "new" in data

    def test_creates_parent_dirs(self, tmp_path):
        ledger = tmp_path / "sub" / "dir" / "seen.json"
        save_seen({}, ledger)
        assert ledger.exists()


class TestFilterUnseen:
    def test_filters_seen(self):
        papers = [_make_paper("p1"), _make_paper("p2"), _make_paper("p3")]
        seen = {"p1": "2025-01-15", "p3": "2025-01-14"}
        result = filter_unseen(papers, seen)
        assert len(result) == 1
        assert result[0].paper_id == "p2"

    def test_all_new(self):
        papers = [_make_paper("p1")]
        result = filter_unseen(papers, {})
        assert len(result) == 1

    def test_all_seen(self):
        papers = [_make_paper("p1")]
        result = filter_unseen(papers, {"p1": "2025-01-15"})
        assert result == []


class TestRecordPapers:
    def test_adds_to_seen(self):
        papers = [_make_paper("p1"), _make_paper("p2")]
        seen: dict[str, str] = {}
        result = record_papers(papers, seen, "2025-01-15")
        assert result == {"p1": "2025-01-15", "p2": "2025-01-15"}
        assert seen is result  # mutates in place
