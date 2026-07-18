"""Reproducibility and environment helpers."""

from __future__ import annotations

import random
import shutil
import subprocess

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def git_revision() -> str | None:
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        return subprocess.run(  # noqa: S603 - shutil-resolved executable, fixed arguments
            [executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
