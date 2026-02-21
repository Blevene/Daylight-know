"""Tests for the email dispatch module."""

from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.emailer import _build_email, send_digest
from digest_pipeline.pipeline import PaperAnalysis


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
        dry_run=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


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


def test_build_email_structure():
    settings = _make_settings()
    papers = _make_papers()
    msg = _build_email(papers, "2025-01-15", settings)
    assert msg["Subject"] == "Research Digest — 2025-01-15"
    assert msg["To"] == "c@d.com"
    payloads = msg.get_payload()
    assert len(payloads) == 2  # plaintext + HTML


def test_build_email_with_implications_and_critiques():
    settings = _make_settings()
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


def test_build_email_without_optional_sections():
    settings = _make_settings()
    papers = [PaperAnalysis(
        title="Test Paper",
        url="https://arxiv.org/abs/2401.00001",
        authors=["Alice"],
        summary="Test summary",
    )]
    msg = _build_email(papers, "2025-01-15", settings)
    payloads = msg.get_payload()
    html_body = payloads[1].get_payload(decode=True).decode()
    assert "Practical Implications" not in html_body
    assert "Critique" not in html_body


def test_build_email_per_paper_structure():
    settings = _make_settings()
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


def test_dry_run_does_not_send(capsys):
    settings = _make_settings(dry_run=True)
    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    captured = capsys.readouterr()
    assert "Test summary" in captured.out


def test_dry_run_includes_implications_and_critiques(capsys):
    settings = _make_settings(dry_run=True)
    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    captured = capsys.readouterr()
    assert "Apply this to X." in captured.out
    assert "Limitation in Y." in captured.out


@patch("digest_pipeline.emailer.smtplib.SMTP_SSL")
def test_real_send(mock_smtp_cls):
    settings = _make_settings(dry_run=False)
    mock_server = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    papers = _make_papers()
    send_digest(papers, "2025-01-15", settings)
    mock_server.login.assert_called_once_with("u", "p")
    mock_server.send_message.assert_called_once()
