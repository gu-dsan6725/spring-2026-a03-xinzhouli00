from __future__ import annotations

import os
import time
from pathlib import Path
from typing import NamedTuple

from .router import classify_query, INTENT_RULES
from .csv_retriever import CSVRetriever
from .text_retriever import TextRetriever

# Re-export so notebook can do: from src.part2.pipeline import classify_query
__all__ = [
    "classify_query", "INTENT_RULES", "TEST_QUESTIONS",
    "answer_question", "run_all_questions",
]

TEST_QUESTIONS = [
    "What was the total revenue for Electronics category in December 2024?",
    "Which region had the highest sales volume?",
    "What are the key features of the Wireless Bluetooth Headphones?",
    "What do customers say about the Air Fryer's ease of cleaning?",
    "Which product has the best customer reviews and how well is it selling?",
    "I want a product for fitness that is highly rated and sells well in the West region. What do you recommend?",
]


class MultiSourceResult(NamedTuple):
    question: str
    tags: list[str]
    csv_context: str
    text_context: str
    answer: str

    def format_for_file(self) -> str:
        lines = [
            f"Question : {self.question}",
            f"Sources  : {self.tags}",
            "",
            "--- CSV Context ---",
            self.csv_context or "(not used)",
            "",
            "--- Text Context ---",
            self.text_context or "(not used)",
            "",
            "--- Answer ---",
            self.answer,
        ]
        return "\n".join(lines)


def _build_prompt(question: str, csv_ctx: str, text_ctx: str) -> str:
    parts = [
        "You are a helpful e-commerce analyst. Answer the question using ONLY the data below.",
        "Cite specific numbers, product names, or file sources where relevant.",
        "",
        f"Question: {question}",
    ]
    if csv_ctx:
        parts += ["", "=== Structured Sales Data (CSV) ===", csv_ctx]
    if text_ctx:
        parts += ["", "=== Product Pages (Text) ===", text_ctx]
    return "\n".join(parts)


def answer_question(
    data_dir: Path,
    question: str,
    max_chars: int = 6000,
) -> MultiSourceResult:
    """Route → retrieve → combine → LLM → MultiSourceResult."""
    csv_path  = data_dir / "structured" / "daily_sales.csv"
    text_dir  = data_dir / "unstructured"

    tags = classify_query(question)
    csv_ctx = text_ctx = ""

    if "csv" in tags and csv_path.exists():
        csv_ctx = CSVRetriever(csv_path, max_chars=max_chars).retrieve(question)

    if "text" in tags and text_dir.exists():
        text_ctx = TextRetriever(text_dir, max_chars=max_chars).retrieve(question)

    prompt = _build_prompt(question, csv_ctx, text_ctx)

    # LLM call — reuse Part 1 LLMClient (same .env keys)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from part1.llm import LLMClient
        llm = LLMClient()
        answer = llm.generate(prompt, max_tokens=1500)
    except Exception as exc:
        answer = f"LLM unavailable — {exc}"

    return MultiSourceResult(
        question=question,
        tags=tags,
        csv_context=csv_ctx,
        text_context=text_ctx,
        answer=answer,
    )


def run_all_questions(data_dir: Path, output_path: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"data/ not found at {data_dir}")

    throttle = float(os.getenv("LLM_THROTTLE_SECONDS", "0"))
    sections: list[str] = []

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        tags = classify_query(q)
        print(f"[{idx}/{len(TEST_QUESTIONS)}] {tags} | {q[:60]}...")
        result = answer_question(data_dir, q)
        sections.append(f"# Question {idx}\n{result.format_for_file()}")
        if throttle > 0 and idx < len(TEST_QUESTIONS):
            time.sleep(throttle)

    output_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    print(f"\n Written → {output_path}")
