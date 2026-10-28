"""Run a command in its own process under a memory / time watchdog.

The NPU compiler runs inside the calling process and can grab tens of GB on some graphs (PLAN.md,
L5 incident): every benchmark job runs in a child process that is killed past `max_gb` of committed
memory, when the system runs low on memory, or past its timeout.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Watched:
    status: str | None  # None if the process ended by itself, else "killed: <reason>"
    returncode: int | None
    lines: list[str] = field(default_factory=list)
    seconds: float = 0.0
    peak_gb: float = 0.0
    on_ac: bool | None = None  # laptop performance depends on AC vs battery


def _family(proc: psutil.Process) -> list[psutil.Process]:
    """The process and its children; empty once it has exited (it may exit at any time)."""
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return []


def _committed_gb(proc: psutil.Process) -> float:
    total = 0
    for p in _family(proc):
        try:
            info = p.memory_info()
            total += getattr(info, "private", info.vms)  # Windows: committed private bytes
        except psutil.Error:
            pass
    return total / 1e9


def run_watched(cmd: list[str], timeout: float, max_gb: float = 12.0, min_free_gb: float = 1.0,
                echo: bool = False) -> Watched:
    """Run `cmd` from the repository root; stdout and stderr are collected (and echoed if asked)."""
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    watched = psutil.Process(proc.pid)
    peak, status = 0.0, None
    lines: list[str] = []

    def read():
        for line in proc.stdout:
            lines.append(line)
            if echo:
                print(line, end="", flush=True)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while proc.poll() is None:
        peak = max(peak, _committed_gb(watched))
        free = psutil.virtual_memory().available / 1e9
        if peak > max_gb:
            status = f"killed: {peak:.1f} GB committed > {max_gb} GB"
        elif free < min_free_gb:
            status = f"killed: system free memory {free:.1f} GB < {min_free_gb} GB"
        elif time.time() - started > timeout:
            status = f"killed: timeout {timeout:.0f} s"
        if status:
            for p in _family(watched):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(0.5)
    proc.wait()
    reader.join(timeout=5)
    battery = psutil.sensors_battery()
    return Watched(status, proc.returncode, lines, round(time.time() - started, 1), round(peak, 2),
                   None if battery is None else battery.power_plugged)
