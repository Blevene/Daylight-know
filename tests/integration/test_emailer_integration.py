"""Integration tests for emailer.py — real SMTP via aiosmtpd + Jinja2 edge cases.

Test IDs: EM-1, EM-2
"""

import email
import email.header
import socket
import time

import pytest
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Message

from digest_pipeline.config import Settings
from digest_pipeline.emailer import _build_email
from digest_pipeline.pipeline import PaperAnalysis


class _CapturingSMTPHandler(Message):
    """aiosmtpd handler that captures received messages."""

    def __init__(self):
        super().__init__()
        self.messages: list[email.message.Message] = []

    def handle_message(self, message):
        self.messages.append(message)


@pytest.mark.integration
@pytest.mark.timeout(15)
class TestEmailerIntegration:
    """Tests that exercise real SMTP sending via a local aiosmtpd server."""

    @staticmethod
    def _free_port():
        """Find a free TCP port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_real_smtp_send(self, tmp_path):
        """EM-1: Send a real email to a local SMTP server, verify receipt."""
        port = self._free_port()
        handler = _CapturingSMTPHandler()
        controller = Controller(handler, hostname="127.0.0.1", port=port)
        controller.start()
        try:
            settings = Settings(
                _env_file=None,
                llm_api_key="k",
                smtp_host="127.0.0.1",
                smtp_port=port,
                smtp_user="",
                smtp_password="",
                email_from="sender@test.com",
                email_to="recipient@test.com",
                dry_run=False,
            )

            import smtplib

            papers = [
                PaperAnalysis(
                    title="Test Paper",
                    url="https://arxiv.org/abs/2401.00001",
                    authors=["Alice", "Bob"],
                    summary="Test summary content",
                    implications="Test implication",
                    critique="Test critique",
                ),
            ]

            msg = _build_email(papers, "2025-01-15", settings)

            with smtplib.SMTP("127.0.0.1", port) as server:
                server.send_message(msg)

            time.sleep(0.2)

            assert len(handler.messages) == 1
            received = handler.messages[0]
            subject = str(email.header.make_header(email.header.decode_header(received["Subject"])))
            assert "Research Digest" in subject
            assert received["From"] == "sender@test.com"
            assert received["To"] == "recipient@test.com"
        finally:
            controller.stop()

    def test_jinja2_template_edge_cases(self):
        """EM-2: Templates handle special characters, empty sections, long summaries."""
        settings = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )

        # Special characters in summary
        papers = [PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/1",
            authors=["Alice"],
            summary='Summary with <html> & "quotes" and unicode: café résumé',
        )]
        msg = _build_email(papers, "2025-01-15", settings)
        payloads = msg.get_payload()
        plain_body = payloads[0].get_payload(decode=True).decode()
        assert "café" in plain_body

        # Empty implications and critiques (sections should be omitted)
        papers2 = [PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/1",
            authors=["Alice"],
            summary="Summary",
        )]
        msg2 = _build_email(papers2, "2025-01-15", settings)
        html_body = msg2.get_payload()[1].get_payload(decode=True).decode()
        assert "Practical Implications" not in html_body

        # Very long summary
        papers3 = [PaperAnalysis(
            title="Test Paper",
            url="https://arxiv.org/abs/1",
            authors=["Alice"],
            summary="Word " * 5000,
        )]
        msg3 = _build_email(papers3, "2025-01-15", settings)
        plain_body3 = msg3.get_payload()[0].get_payload(decode=True).decode()
        assert len(plain_body3) > 20000
