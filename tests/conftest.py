"""Shared pytest configuration and markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external dependencies")
    config.addinivalue_line("markers", "integration: real local dependencies (ChromaDB, Chonkie, PyMuPDF)")
    config.addinivalue_line("markers", "e2e: full pipeline runs")
    config.addinivalue_line("markers", "network: requires internet access (arXiv, GitHub)")
