"""
Pytest configuration ensuring project root is in Python path for test discovery.
"""

import os
import sys
from pathlib import Path

# Insert project root into sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure local HuggingFace cache is discovered
cache_dir = root_dir / ".cache" / "huggingface"
if cache_dir.exists():
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

@pytest.fixture(autouse=True)
def hermetic_test_env(monkeypatch):
    """Ensure automated pytest tests run deterministically and offline without network flakiness."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
