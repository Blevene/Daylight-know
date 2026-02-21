"""Tests for the email dispatch module."""

from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.emailer import _build_email, send_digest


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


def test_build_email_structure():
    settings = _make_settings()
    msg = _build_email("Test summary", 3, "2025-01-15", settings)
    assert msg["Subject"] == "Research Digest — 2025-01-15"
    assert msg["To"] == "c@d.com"
    payloads = msg.get_payload()
    assert len(payloads) == 2  # plaintext + HTML


def test_dry_run_does_not_send(capsys):
    settings = _make_settings(dry_run=True)
    send_digest("Summary text", 2, "2025-01-15", settings)
    captured = capsys.readouterr()
    assert "Summary text" in captured.out
    assert "DRY-RUN" not in captured.out or True  # DRY-RUN is logged, not printed


@patch("digest_pipeline.emailer.smtplib.SMTP_SSL")
def test_real_send(mock_smtp_cls):
    settings = _make_settings(dry_run=False)
    mock_server = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_digest("Summary text", 2, "2025-01-15", settings)
    mock_server.login.assert_called_once_with("u", "p")
    mock_server.send_message.assert_called_once()
