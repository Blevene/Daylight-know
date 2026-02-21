"""Centralised configuration via environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All pipeline-level settings.

    Values are read from environment variables (case-insensitive) and,
    optionally, from a ``.env`` file located in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── arXiv ───────────────────────────────────────────────────
    arxiv_topics: list[str] = Field(default=["cs.AI", "cs.LG"])
    arxiv_max_results: int = Field(default=50)

    # ── LLM (via litellm — supports any provider) ─────────────
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="openai/gpt-4o-mini")
    llm_max_tokens: int = Field(default=4096)
    llm_api_base: str | None = Field(default=None)

    # ── ChromaDB ────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=Path("./data/chromadb"))
    chroma_collection: str = Field(default="research_digest")

    # ── Email / SMTP ────────────────────────────────────────────
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=465)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    email_from: str = Field(default="")
    email_to: str = Field(default="")

    # ── Pipeline mode ───────────────────────────────────────────
    dry_run: bool = Field(default=True)

    # ── GitHub (optional) ───────────────────────────────────────
    github_enabled: bool = Field(default=False)
    github_languages: list[str] = Field(default=["python"])
    github_top_n: int = Field(default=5)

    # ── Post-processing (optional) ──────────────────────────────
    postprocessing_implications: bool = Field(default=True)
    postprocessing_critiques: bool = Field(default=True)

    # ── PDF download retry ──────────────────────────────────────
    pdf_download_max_retries: int = Field(default=3)


def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    return Settings()
