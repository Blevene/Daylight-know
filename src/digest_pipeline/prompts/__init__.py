from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt from ``prompts/{name}.md``.

    Raises ``FileNotFoundError`` with a helpful message listing available
    prompt files if the requested name does not exist.
    """
    path = _DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in _DIR.glob("*.md"))
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Available prompts: {available}"
        )
    return path.read_text().strip()
