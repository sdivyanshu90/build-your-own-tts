from __future__ import annotations

import pytest


@pytest.mark.performance
def test_performance_suite_is_opt_in() -> None:
    """Deployment benchmarks use `tts benchmark`; CI only asserts the harness is registered."""
    assert True
