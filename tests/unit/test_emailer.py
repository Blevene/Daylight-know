"""Tests for the email dispatch module."""

from unittest.mock import MagicMock, patch

from digest_pipeline.emailer import _build_email, send_digest
from digest_pipeline.pipeline import PaperAnalysis


def _make_papers():
    return [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice", "Bob"],
            summary="Test summary",
            implications="Apply this to X.",
            critique="Limitation in Y.",
        ),
    ]


def test_build_email_structure(make_settings):
    settings = make_settings()
    papers = _make_papers()
    msg = _build_email(papers, "2025-01-15", settings)
    assert msg["Subject"] == "Research Digest — 2025-01-15"
    assert msg["To"] == "c@d.com"
    payloads = msg.get_payload()
    assert len(payloads) == 2  # plaintext + HTML


def test_build_email_with_implications_and_critiques(make_settings):
    settings = make_settings()
    papers = _make_papers()
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "Practical Implications" in html_body
    assert "Apply this to X." in html_body
    assert "Critique" in html_body
    assert "Limitation in Y." in html_body
    assert "Practical Implications" in plain_body
    assert "Apply this to X." in plain_body


def test_build_email_without_optional_sections(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
        )
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    assert "Practical Implications" not in html_body
    assert "Critique" not in html_body


def test_build_email_per_paper_structure(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Paper One",
            url="https://arxiv.org/abs/1",
            authors=["Alice"],
            summary="Summary one",
            implications="Impl one",
            critique="Crit one",
        ),
        PaperAnalysis(
            title="Paper Two",
            url="https://arxiv.org/abs/2",
            authors=["Bob"],
            summary="Summary two",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    # Both papers present
    assert "Paper One" in html_body
    assert "Paper Two" in html_body
    # Links present
    assert 'href="https://arxiv.org/abs/1"' in html_body
    assert 'href="https://arxiv.org/abs/2"' in html_body
    # Paper count
    assert "2 paper(s)" in html_body


def test_build_email_with_categories(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            categories=["cs.AI", "cs.LG"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "cs.AI" in html_body
    assert "cs.LG" in html_body
    assert "cs.AI · cs.LG" in html_body
    assert "Categories: cs.AI, cs.LG" in plain_body


def test_build_email_without_categories(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert '<p class="categories">' not in html_body
    assert "Categories:" not in plain_body


def test_build_email_mixed_categories(make_settings):
    """One paper with categories, one without — both render correctly."""
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Paper With Cats",
            url="https://arxiv.org/abs/1",
            authors=["Alice"],
            categories=["cs.AI", "cs.LG"],
            summary="Summary one",
        ),
        PaperAnalysis(
            title="Paper Without Cats",
            url="https://arxiv.org/abs/2",
            authors=["Bob"],
            summary="Summary two",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    # First paper has categories
    assert "cs.AI · cs.LG" in html_body
    assert "Categories: cs.AI, cs.LG" in plain_body
    # Second paper does not
    assert html_body.count('<p class="categories">') == 1
    assert plain_body.count("Categories:") == 1
    # Both papers present
    assert "Paper With Cats" in html_body
    assert "Paper Without Cats" in html_body


def test_build_email_with_upvotes(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="HF Paper",
            url="https://huggingface.co/papers/2401.00001",
            source="huggingface",
            authors=["Alice"],
            upvotes=42,
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "42 upvotes" in html_body
    assert "upvote-badge" in html_body
    assert "Upvotes: 42" in plain_body


def test_build_email_without_upvotes(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "0 upvotes" not in html_body
    assert "Upvotes:" not in plain_body


def test_build_email_with_fields_of_study(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="OA Paper",
            url="https://openalex.org/W123",
            source="openalex",
            authors=["Bob"],
            fields_of_study=["Computer Science", "Mathematics"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    assert "Computer Science" in html_body
    assert "Mathematics" in html_body
    assert "field-tag" in html_body
    assert "Fields of Study: Computer Science, Mathematics" in plain_body


def test_build_email_without_fields_of_study(make_settings):
    settings = make_settings()
    papers = [
        PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/2401.00001",
            authors=["Alice"],
            summary="Test summary",
        ),
    ]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    plain_body = payloads[0].get_payload(decode=True).decode()
    # The field-tag class only appears in <style>; no actual field tags rendered
    assert 'class="field-tag"' not in html_body
    assert "Fields of Study:" not in plain_body


def test_dry_run_does_not_send(make_settings, capsys):
    settings = make_settings(dry_run=True)
    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    captured = capsys.readouterr()
    assert "Test summary" in captured.out


def test_dry_run_includes_implications_and_critiques(make_settings, capsys):
    settings = make_settings(dry_run=True)
    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    captured = capsys.readouterr()
    assert "Apply this to X." in captured.out
    assert "Limitation in Y." in captured.out


@patch("digest_pipeline.emailer.smtplib.SMTP")
def test_real_send(mock_smtp_cls, make_settings):
    settings = make_settings(dry_run=False)
    mock_server = MagicMock()
    mock_smtp_cls.return_value = mock_server
    mock_server.__enter__ = MagicMock(return_value=mock_server)
    mock_server.__exit__ = MagicMock(return_value=False)

    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("u", "p")
    mock_server.send_message.assert_called_once()
