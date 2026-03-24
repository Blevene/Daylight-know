"""Interactive setup wizard for configuring the digest pipeline.

Guides users through configuring arXiv topics, LLM provider, SMTP email,
ChromaDB storage, and optional features, then writes a .env file.

    digest-pipeline setup
"""

from __future__ import annotations

import shutil
import smtplib
import ssl
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from digest_pipeline.arxiv_topics import (
    GROUPS,
    list_group,
    search_topics,
    validate_topics,
)
from digest_pipeline.openalex_fetcher import OPENALEX_FIELDS

console = Console()

# ── Utility helpers ──────────────────────────────────────────────


def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Prompt for input with an optional default value."""
    suffix = f" [{default}]" if default else ""
    try:
        value = console.input(f"  {label}{suffix}: ", password=secret)
    except EOFError:
        value = ""
    return value.strip() or default


def _prompt_bool(label: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    try:
        value = console.input(f"  {label} [{hint}]: ").strip().lower()
    except EOFError:
        value = ""
    if not value:
        return default
    return value in ("y", "yes")


def _prompt_choice(label: str, choices: list[str], default: str = "") -> str:
    """Numbered list picker."""
    console.print(f"\n  [bold]{label}[/]")
    for i, choice in enumerate(choices, 1):
        marker = " [dim](default)[/]" if choice == default else ""
        console.print(f"    {i}. {choice}{marker}")

    try:
        value = console.input("  Enter number or value: ").strip()
    except EOFError:
        value = ""

    if not value:
        return default

    try:
        idx = int(value) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except ValueError:
        pass

    return value


def _print_header(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]── {title} ──[/]\n")


def _print_info(msg: str) -> None:
    """Print an informational message."""
    console.print(f"  [dim]{msg}[/]")


# ── Interactive topic selector ───────────────────────────────────


def _collect_arxiv_topics() -> list[str]:
    """Interactive topic selection loop."""
    selected: list[str] = []

    console.print("\n  [bold]Select arXiv topics for your digest:[/]")
    console.print("  [dim]You can browse groups, search, or type codes directly.[/]\n")

    while True:
        console.print("  [cyan][b][/] Browse by group    [cyan][s][/] Search by keyword")
        console.print("  [cyan][t][/] Type codes         [cyan][d][/] Done")
        if selected:
            console.print(f"  [green]Selected so far:[/] {', '.join(selected)}")

        try:
            choice = console.input("\n  Action: ").strip().lower()
        except EOFError:
            break

        if choice == "b":
            _browse_groups(selected)
        elif choice == "s":
            _search_and_add(selected)
        elif choice == "t":
            _type_codes(selected)
        elif choice == "d":
            break
        else:
            console.print("  [yellow]Invalid choice. Use b/s/t/d.[/]")

    return selected


def _browse_groups(selected: list[str]) -> None:
    """Browse topic groups and add codes."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, group in enumerate(GROUPS, 1):
        count = len(list_group(group))
        table.add_row(f"  {i}.", f"[cyan]{group}[/]", f"({count} topics)")
    console.print(table)

    try:
        pick = console.input("  Group number: ").strip()
    except EOFError:
        return

    try:
        idx = int(pick) - 1
        if 0 <= idx < len(GROUPS):
            group = GROUPS[idx]
        else:
            console.print("  [yellow]Invalid number.[/]")
            return
    except ValueError:
        console.print("  [yellow]Invalid input.[/]")
        return

    topics = list_group(group)
    topic_table = Table(title=f"Topics in '{group}'", show_lines=False)
    topic_table.add_column("#", style="dim", justify="right")
    topic_table.add_column("Code", style="cyan bold")
    topic_table.add_column("Name")

    for i, t in enumerate(topics, 1):
        marker = " [green]✓[/]" if t.code in selected else ""
        topic_table.add_row(str(i), t.code, f"{t.name}{marker}")

    console.print(topic_table)

    try:
        picks = console.input("  Add topics (numbers, comma-separated, or 'all'): ").strip()
    except EOFError:
        return

    _add_picks(picks, topics, selected)


def _add_picks(picks: str, items: list, selected: list[str]) -> None:
    """Parse comma-separated picks (numbers or 'all') and add to selected."""
    if picks.lower() == "all":
        for t in items:
            if t.code not in selected:
                selected.append(t.code)
        return

    for p in picks.split(","):
        p = p.strip()
        try:
            idx = int(p) - 1
            if 0 <= idx < len(items) and items[idx].code not in selected:
                selected.append(items[idx].code)
        except ValueError:
            pass


def _search_and_add(selected: list[str]) -> None:
    """Search topics and add matches."""
    try:
        query = console.input("  Search query: ").strip()
    except EOFError:
        return

    if not query:
        return

    results = search_topics(query)
    if not results:
        console.print(f"  [yellow]No topics found for '{query}'.[/]")
        return

    for i, t in enumerate(results, 1):
        marker = " [green]✓[/]" if t.code in selected else ""
        console.print(f"    {i}. [cyan]{t.code}[/] — {t.name}{marker}")

    try:
        picks = console.input("  Add topics (numbers, comma-separated, or 'all'): ").strip()
    except EOFError:
        return

    _add_picks(picks, results, selected)


def _type_codes(selected: list[str]) -> None:
    """Manually type topic codes."""
    try:
        raw = console.input("  Enter codes (comma-separated): ").strip()
    except EOFError:
        return

    if not raw:
        return

    codes = [c.strip() for c in raw.split(",") if c.strip()]
    valid, invalid = validate_topics(codes)

    for code in valid:
        if code not in selected:
            selected.append(code)
            console.print(f"    [green]✓[/] Added {code}")

    for code in invalid:
        console.print(f"    [red]✗[/] Invalid code: {code}")


# ── OpenAlex field selector ───────────────────────────────────


def _collect_openalex_fields() -> list[str]:
    """Interactive multi-select for OpenAlex academic fields."""
    field_names = list(OPENALEX_FIELDS.keys())

    console.print("\n  [bold]Select OpenAlex fields to filter by:[/]")
    console.print("  [dim]Enter numbers (comma-separated), 'all', or press Enter to skip.[/]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    half = (len(field_names) + 1) // 2
    for i in range(half):
        left = f"  {i + 1:>2}. {field_names[i]}"
        right_idx = i + half
        if right_idx < len(field_names):
            right = f"{right_idx + 1:>2}. {field_names[right_idx]}"
        else:
            right = ""
        table.add_row(left, right)

    console.print(table)

    try:
        picks = console.input("\n  Fields (numbers, 'all', or Enter to skip): ").strip()
    except EOFError:
        return []

    if not picks:
        return []

    if picks.lower() == "all":
        return field_names

    selected: list[str] = []
    for p in picks.split(","):
        p = p.strip()
        try:
            idx = int(p) - 1
            if 0 <= idx < len(field_names):
                name = field_names[idx]
                if name not in selected:
                    selected.append(name)
                    console.print(f"    [green]✓[/] {name}")
            else:
                console.print(f"    [yellow]⚠ Invalid number: {p}[/]")
        except ValueError:
            match = [f for f in field_names if f.lower() == p.lower()]
            if match and match[0] not in selected:
                selected.append(match[0])
                console.print(f"    [green]✓[/] {match[0]}")
            else:
                console.print(f"    [yellow]⚠ Unknown field: {p}[/]")

    return selected


# ── Config collectors ────────────────────────────────────────────


def _collect_arxiv_settings() -> dict[str, str]:
    """Collect arXiv configuration."""
    _print_header("arXiv Settings")

    topics = _collect_arxiv_topics()
    if not topics:
        topics = ["cs.AI", "cs.LG"]
        console.print(f"  [yellow]No topics selected, using defaults: {', '.join(topics)}[/]")

    max_results = _prompt("Max results per topic", "50")

    return {
        "ARXIV_TOPICS": ",".join(topics),
        "ARXIV_MAX_RESULTS": max_results,
    }


def _collect_llm_settings() -> dict[str, str]:
    """Collect LLM provider configuration."""
    _print_header("LLM Settings")
    _print_info("Uses litellm — supports OpenAI, Anthropic, Ollama, Azure, etc.")
    _print_info('Format: "provider/model-name" (e.g. openai/gpt-4o-mini)')

    model = _prompt("Model", "openai/gpt-4o-mini")
    api_key = _prompt("API key", secret=True)
    max_tokens = _prompt("Max tokens", "4096")
    api_base = _prompt("API base URL (optional, press Enter to skip)")

    config = {
        "LLM_MODEL": model,
        "LLM_API_KEY": api_key,
        "LLM_MAX_TOKENS": max_tokens,
    }
    if api_base:
        config["LLM_API_BASE"] = api_base

    return config


def _collect_smtp_settings() -> dict[str, str]:
    """Collect SMTP email configuration."""
    _print_header("Email / SMTP Settings")

    host = _prompt("SMTP host", "smtp.gmail.com")
    port = _prompt("SMTP port", "587")
    user = _prompt("SMTP user (email)")
    password = _prompt("SMTP password", secret=True)
    email_from = _prompt("From address", default=user)
    email_to = _prompt("To address (recipient)")

    return {
        "SMTP_HOST": host,
        "SMTP_PORT": port,
        "SMTP_USER": user,
        "SMTP_PASSWORD": password,
        "EMAIL_FROM": email_from,
        "EMAIL_TO": email_to,
    }


def _collect_chroma_settings() -> dict[str, str]:
    """Collect ChromaDB configuration."""
    _print_header("ChromaDB Settings")

    persist_dir = _prompt("Persist directory", "./data/chromadb")
    collection = _prompt("Collection name", "research_digest")

    return {
        "CHROMA_PERSIST_DIR": persist_dir,
        "CHROMA_COLLECTION": collection,
    }


def _collect_optional_settings() -> dict[str, str]:
    """Collect optional feature toggles."""
    _print_header("Optional Features")

    config: dict[str, str] = {}

    config["DRY_RUN"] = "true" if _prompt_bool("Start in dry-run mode?", default=True) else "false"

    config["POSTPROCESSING_IMPLICATIONS"] = (
        "true" if _prompt_bool("Enable implications extraction?", default=True) else "false"
    )
    config["POSTPROCESSING_CRITIQUES"] = (
        "true" if _prompt_bool("Enable critique generation?", default=True) else "false"
    )

    config["PDF_DOWNLOAD_MAX_RETRIES"] = _prompt("PDF download max retries", "3")
    config["PDF_DOWNLOAD_WORKERS"] = _prompt("Parallel PDF download workers", "8")
    archive_dir = _prompt("PDF archive directory (optional, press Enter to skip)")
    if archive_dir:
        config["PDF_ARCHIVE_DIR"] = archive_dir

    if _prompt_bool("Enable HuggingFace Daily Papers?", default=False):
        config["HUGGINGFACE_ENABLED"] = "true"
        config["HUGGINGFACE_TOKEN"] = _prompt("HuggingFace token (optional, press Enter to skip)")
        config["HUGGINGFACE_MAX_RESULTS"] = _prompt("HuggingFace max results", "20")
    else:
        config["HUGGINGFACE_ENABLED"] = "false"

    if _prompt_bool("Enable OpenAlex integration?", default=False):
        config["OPENALEX_ENABLED"] = "true"
        config["OPENALEX_API_KEY"] = _prompt("OpenAlex API key (optional, press Enter to skip)")
        config["OPENALEX_EMAIL"] = _prompt("Email for OpenAlex polite pool (optional)")
        config["OPENALEX_MAX_RESULTS"] = _prompt("OpenAlex max results", "20")
        config["OPENALEX_QUERY"] = _prompt("OpenAlex search query", "machine learning")
        import json as _json

        selected_fields = _collect_openalex_fields()
        if selected_fields:
            config["OPENALEX_FIELDS"] = _json.dumps(selected_fields)
        # Interest-based ranking (pipeline-wide — applies to arXiv + OpenAlex)
        if _prompt_bool("Enable interest-based paper ranking?", default=False):
            config["INTEREST_PROFILE"] = _prompt(
                "Describe your research interests (natural language)"
            )
            keywords = _prompt("Boost keywords (comma-separated, optional)")
            if keywords:
                kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
                config["INTEREST_KEYWORDS"] = _json.dumps(kw_list)
            config["ARXIV_FETCH_POOL"] = _prompt("arXiv papers to fetch before ranking", "200")
    else:
        config["OPENALEX_ENABLED"] = "false"

    if _prompt_bool("Enable GitHub trending integration?", default=False):
        config["GITHUB_ENABLED"] = "true"
        config["GITHUB_LANGUAGES"] = _prompt("GitHub languages (comma-separated)", "python,rust")
        config["GITHUB_TOP_N"] = _prompt("GitHub top N repos", "5")
    else:
        config["GITHUB_ENABLED"] = "false"

    return config


# ── Live validators ──────────────────────────────────────────────


def _test_llm_connection(config: dict[str, str]) -> bool:
    """Test LLM connection with a tiny prompt."""
    try:
        import litellm
    except ImportError:
        console.print("  [red]✗ litellm is not installed. Run: pip install litellm[/]")
        return False

    try:
        kwargs: dict = {
            "model": config["LLM_MODEL"],
            "messages": [{"role": "user", "content": "Say 'ok'."}],
            "max_tokens": 10,
        }
        if config.get("LLM_API_KEY"):
            kwargs["api_key"] = config["LLM_API_KEY"]
        if config.get("LLM_API_BASE"):
            kwargs["api_base"] = config["LLM_API_BASE"]

        litellm.completion(**kwargs)
        console.print("  [green]✓ LLM connection successful![/]")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ LLM connection failed: {exc}[/]")
        return False


def _test_smtp_connection(config: dict[str, str]) -> bool:
    """Test SMTP connection with STARTTLS and login."""
    try:
        host = config["SMTP_HOST"]
        port = int(config["SMTP_PORT"])
        user = config["SMTP_USER"]
        password = config["SMTP_PASSWORD"]

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(user, password)

        console.print("  [green]✓ SMTP connection successful![/]")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ SMTP connection failed: {exc}[/]")
        return False


# ── .env file handling ───────────────────────────────────────────


def _read_existing_env(path: Path) -> dict[str, str]:
    """Parse an existing .env file into a dict."""
    result: dict[str, str] = {}
    if not path.exists():
        return result

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            val = value.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            result[key.strip()] = val

    return result


def _write_env_file(config: dict[str, str], path: Path) -> None:
    """Write config to a .env file with grouped sections and comments."""
    sections = [
        (
            "arXiv Settings",
            ["ARXIV_TOPICS", "ARXIV_MAX_RESULTS"],
        ),
        (
            "LLM Settings (via litellm — supports any provider)",
            ["LLM_API_KEY", "LLM_MODEL", "LLM_MAX_TOKENS", "LLM_API_BASE"],
        ),
        (
            "ChromaDB Settings",
            ["CHROMA_PERSIST_DIR", "CHROMA_COLLECTION"],
        ),
        (
            "Email / SMTP Settings",
            ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO"],
        ),
        (
            "Pipeline Modes",
            ["DRY_RUN"],
        ),
        (
            "Post-processing",
            ["POSTPROCESSING_IMPLICATIONS", "POSTPROCESSING_CRITIQUES"],
        ),
        (
            "PDF Download",
            ["PDF_DOWNLOAD_MAX_RETRIES", "PDF_DOWNLOAD_WORKERS"],
        ),
        (
            "PDF Archive",
            ["PDF_ARCHIVE_DIR"],
        ),
        (
            "Optional: HuggingFace Daily Papers",
            ["HUGGINGFACE_ENABLED", "HUGGINGFACE_TOKEN", "HUGGINGFACE_MAX_RESULTS"],
        ),
        (
            "Interest-Based Ranking",
            ["INTEREST_PROFILE", "INTEREST_KEYWORDS", "ARXIV_FETCH_POOL"],
        ),
        (
            "Optional: OpenAlex",
            [
                "OPENALEX_ENABLED",
                "OPENALEX_API_KEY",
                "OPENALEX_EMAIL",
                "OPENALEX_MAX_RESULTS",
                "OPENALEX_QUERY",
                "OPENALEX_FIELDS",
            ],
        ),
        (
            "Optional: GitHub Trending",
            ["GITHUB_ENABLED", "GITHUB_LANGUAGES", "GITHUB_TOP_N"],
        ),
    ]

    def _format_env_line(key: str, value: str) -> str:
        """Format a key=value line; skip extra quoting for JSON arrays."""
        if value.startswith("["):
            return f"{key}={value}"
        return f'{key}="{value}"'

    lines: list[str] = []
    written_keys: set[str] = set()

    for title, keys in sections:
        section_lines: list[str] = []
        for key in keys:
            if key in config:
                section_lines.append(_format_env_line(key, config[key]))
                written_keys.add(key)

        if section_lines:
            lines.append(f"# ── {title} {'─' * max(1, 60 - len(title))}")
            lines.extend(section_lines)
            lines.append("")

    # Write any remaining keys not in known sections
    remaining = {k: v for k, v in config.items() if k not in written_keys}
    if remaining:
        lines.append("# ── Other Settings ─────────────────────────────────────────")
        for key, value in remaining.items():
            lines.append(_format_env_line(key, value))
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _handle_existing_env(path: Path, new_config: dict[str, str]) -> dict[str, str]:
    """Handle existing .env file — offer overwrite or merge."""
    if not path.exists():
        return new_config

    console.print(f"\n  [yellow]Existing .env file found at {path}[/]")
    existing = _read_existing_env(path)

    choice = _prompt_choice(
        "How to handle existing .env?",
        ["Overwrite (backup old file)", "Merge (keep existing, add new)"],
        default="Overwrite (backup old file)",
    )

    # Create timestamped backup
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_path = path.parent / f"{path.name}.bak.{timestamp}"
    shutil.copy2(path, backup_path)
    console.print(f"  [dim]Backup saved to {backup_path}[/]")

    if "Merge" in choice:
        merged = dict(existing)
        merged.update(new_config)
        return merged

    return new_config


# ── Orchestrator ─────────────────────────────────────────────────


def run_setup_wizard(env_path: str | None = None) -> None:
    """Run the interactive setup wizard."""
    path = Path(env_path) if env_path else Path(".env")

    console.print(
        Panel(
            "[bold]Welcome to the Digest Pipeline Setup Wizard![/]\n\n"
            "This wizard will guide you through configuring your\n"
            "research digest pipeline (arXiv, HuggingFace, OpenAlex).\n\n"
            "[dim]Press Enter to accept default values shown in brackets.[/]",
            title="🔬 Digest Pipeline Setup",
            border_style="cyan",
        )
    )

    # Collect all settings
    config: dict[str, str] = {}
    config.update(_collect_arxiv_settings())

    llm_config = _collect_llm_settings()
    config.update(llm_config)

    # Test LLM connection
    if _prompt_bool("Test LLM connection now?", default=True):
        _test_llm_connection(llm_config)

    smtp_config = _collect_smtp_settings()
    config.update(smtp_config)

    # Test SMTP connection
    if _prompt_bool("Test SMTP connection now?", default=True):
        _test_smtp_connection(smtp_config)

    config.update(_collect_chroma_settings())
    config.update(_collect_optional_settings())

    # Handle existing .env
    config = _handle_existing_env(path, config)

    # Write .env
    _write_env_file(config, path)
    console.print(f"\n  [green]✓ Configuration written to {path}[/]")

    # Summary
    summary = Table(title="Configuration Summary", show_lines=False)
    summary.add_column("Setting", style="cyan")
    summary.add_column("Value", style="white")

    for key, value in config.items():
        display = "****" if "PASSWORD" in key or "API_KEY" in key or "TOKEN" in key else value
        summary.add_row(key, display)

    console.print(summary)

    # Offer dry-run
    if _prompt_bool("\nRun a dry-run to verify configuration?", default=False):
        console.print("\n  [dim]Running dry-run verification...[/]\n")
        from digest_pipeline.config import get_settings
        from digest_pipeline.pipeline import run

        settings = get_settings()
        settings.dry_run = True
        run(settings)
