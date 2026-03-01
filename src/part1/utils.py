from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import List


def compact_text(text: str, max_chars: int) -> str:
    """Collapse 3+ blank lines to 2, then hard-truncate to max_chars."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class QAResult:
    question: str
    answer: str
    context: str
    commands: List[str] = field(default_factory=list)

    def format_for_file(self) -> str:
        cmd_block = "\n".join(f"  $ {c}" for c in self.commands)
        return dedent(f"""
            ## Question
            {self.question}

            ## Commands Run
            {cmd_block}

            ## Answer
            {self.answer}
        """)
