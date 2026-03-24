# Contributing to Daylight-know

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Blevene/Daylight-know.git
cd Daylight-know
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests (fast, no external deps)
pytest tests/unit/ -q

# Integration tests (requires local ChromaDB, pypdf)
pytest tests/integration/ -q

# All tests
pytest
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (Python 3.10, 100-char line length).

```bash
ruff check .
ruff format .
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass (`pytest tests/unit/ -q`)
4. Run the linter (`ruff check .`)
5. Open a pull request with a clear description of the change

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.
