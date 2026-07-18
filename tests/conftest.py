from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tts_pipeline.config import load_settings  # noqa: E402


@pytest.fixture
def settings():
    return load_settings(Path(__file__).parents[1] / "configs/test.yaml")
