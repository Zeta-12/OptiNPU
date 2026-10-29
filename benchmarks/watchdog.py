"""Run a command in its own process under a memory / time watchdog (now optinpu.sandbox).

Kept for the M0 / M1 runners, which run their jobs from the repository root.
"""

from __future__ import annotations

from pathlib import Path

from optinpu.sandbox import Watched
from optinpu.sandbox import run_watched as _run_watched

ROOT = Path(__file__).resolve().parents[1]


def run_watched(cmd: list[str], timeout: float, max_gb: float = 12.0, min_free_gb: float = 1.0,
                echo: bool = False) -> Watched:
    """Run `cmd` from the repository root; stdout and stderr are collected (and echoed if asked)."""
    return _run_watched(cmd, timeout, max_gb, min_free_gb, echo, cwd=ROOT)
