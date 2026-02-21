"""Tests for the arXiv topics index."""

from digest_pipeline.arxiv_topics import (
    GROUPS,
    TOPICS,
    get_topic,
    is_valid_topic,
    list_group,
    search_topics,
    validate_topics,
)


def test_topics_not_empty():
    assert len(TOPICS) > 100


def test_groups_include_cs():
    assert "cs" in GROUPS


def test_get_topic_known():
    t = get_topic("cs.AI")
    assert t is not None
    assert t.name == "Artificial Intelligence"
    assert t.group == "cs"


def test_get_topic_unknown():
    assert get_topic("xx.ZZ") is None


def test_is_valid_topic():
    assert is_valid_topic("cs.LG") is True
    assert is_valid_topic("fake.XX") is False


def test_list_group_cs():
    cs_topics = list_group("cs")
    assert len(cs_topics) >= 30
    codes = {t.code for t in cs_topics}
    assert "cs.AI" in codes
    assert "cs.LG" in codes


def test_search_topics_by_name():
    results = search_topics("machine learning")
    codes = {t.code for t in results}
    assert "cs.LG" in codes
    assert "stat.ML" in codes


def test_search_topics_by_code():
    results = search_topics("cs.CV")
    assert len(results) >= 1
    assert results[0].code == "cs.CV"


def test_validate_topics():
    valid, invalid = validate_topics(["cs.AI", "fake.XX", "stat.ML"])
    assert valid == ["cs.AI", "stat.ML"]
    assert invalid == ["fake.XX"]
