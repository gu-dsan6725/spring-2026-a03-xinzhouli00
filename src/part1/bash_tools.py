from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .utils import compact_text


@dataclass
class BashResult:
    command: str
    output: str
    exit_code: int


class BashToolRunner:
    def __init__(self, cwd: str | Path, max_chars: int = 8000):
        self.cwd = str(cwd)
        self.max_chars = max_chars

    def run(self, command: str) -> BashResult:
        proc = subprocess.run(
            command,
            cwd=self.cwd,
            shell=True,
            capture_output=True,
            text=True,
        )
        output = proc.stdout.strip()
        if proc.stderr.strip():
            output = (output + "\n" + proc.stderr.strip()).strip()
        output = compact_text(output, self.max_chars)
        return BashResult(command=command, output=output, exit_code=proc.returncode)


@dataclass
class CommandPlan:
    commands: List[str]

    def render(self) -> str:
        return "\n".join(self.commands)


def safe_rg(query: str, globs: List[str] | None = None) -> str:
    parts = ["rg", "-n", shlex.quote(query)]
    if globs:
        for glob in globs:
            parts.append(f"-g {shlex.quote(glob)}")
    return " ".join(parts)


def build_search_cmd(
    pattern: str,
    globs: List[str] | None,
    use_rg: bool,
    roots: List[str] | None = None,
    limit: int | None = None,
) -> str:
    roots = roots or ["."]
    if use_rg:
        cmd = safe_rg(pattern, globs)
        base = f"{cmd} {' '.join(roots)}"
        if limit:
            return f"{base} | head -n {limit}"
        return base
    # grep fallback
    includes = ""
    if globs:
        includes = " ".join(f"--include={shlex.quote(g)}" for g in globs)
    root_str = " ".join(roots)
    base = f"grep -RIn {shlex.quote(pattern)} {includes} {root_str}"
    if limit:
        return f"{base} | head -n {limit}"
    return base
