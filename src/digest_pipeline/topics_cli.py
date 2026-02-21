"""Topic browser CLI — thin adapter over arxiv_topics.py.

Provides subcommands for browsing, searching, and validating arXiv topics:

    digest-pipeline topics list
    digest-pipeline topics search "machine learning"
    digest-pipeline topics group cs
    digest-pipeline topics validate cs.AI cs.LG
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from digest_pipeline.arxiv_topics import (
    GROUPS,
    list_group,
    search_topics,
    validate_topics,
)

console = Console()


def handle_topics_command(args) -> None:
    """Dispatch to the appropriate topics subcommand."""
    cmd = getattr(args, "topics_command", None)
    if cmd is None:
        console.print("[yellow]Usage: digest-pipeline topics {list|search|group|validate}[/]")
        return

    dispatch = {
        "list": _cmd_list,
        "search": lambda: _cmd_search(args.query),
        "group": lambda: _cmd_group(args.name),
        "validate": lambda: _cmd_validate(args.codes),
    }
    handler = dispatch.get(cmd)
    if handler is None:
        console.print(f"[red]Unknown topics subcommand: {cmd}[/]")
        return
    handler()


def _cmd_list() -> None:
    """Show all groups with topic counts."""
    table = Table(title="arXiv Topic Groups", show_lines=False)
    table.add_column("Group", style="cyan bold")
    table.add_column("Topics", justify="right", style="green")
    table.add_column("Example Codes", style="dim")

    for group in GROUPS:
        topics = list_group(group)
        examples = ", ".join(t.code for t in topics[:3])
        if len(topics) > 3:
            examples += ", …"
        table.add_row(group, str(len(topics)), examples)

    console.print(table)


def _cmd_search(query: str) -> None:
    """Search topics by keyword."""
    results = search_topics(query)
    if not results:
        console.print(f"[yellow]No topics found matching '{query}'.[/]")
        return

    table = Table(title=f"Search results for '{query}'")
    table.add_column("Code", style="cyan bold")
    table.add_column("Name", style="white")
    table.add_column("Group", style="dim")

    for t in results:
        table.add_row(t.code, t.name, t.group)

    console.print(table)


def _cmd_group(name: str) -> None:
    """List all topics in a group."""
    topics = list_group(name)
    if not topics:
        console.print(f"[red]Unknown group '{name}'.[/]")
        console.print(f"[dim]Available groups: {', '.join(GROUPS)}[/]")
        return

    table = Table(title=f"Topics in '{name}'")
    table.add_column("Code", style="cyan bold")
    table.add_column("Name", style="white")

    for t in topics:
        table.add_row(t.code, t.name)

    console.print(table)


def _cmd_validate(codes: list[str]) -> None:
    """Validate topic codes."""
    valid, invalid = validate_topics(codes)

    if valid:
        console.print(Panel(
            "\n".join(f"  [green]✓[/] {c}" for c in valid),
            title="Valid Topics",
            border_style="green",
        ))

    if invalid:
        console.print(Panel(
            "\n".join(f"  [red]✗[/] {c}" for c in invalid),
            title="Invalid Topics",
            border_style="red",
        ))
        sys.exit(1)
    else:
        console.print("[green]All topic codes are valid.[/]")
