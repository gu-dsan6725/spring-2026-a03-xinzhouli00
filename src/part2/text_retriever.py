from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple


class TextChunk(NamedTuple):
    product_id: str
    filepath: Path
    content: str
    score: float = 0.0


class TextRetriever:
    def __init__(self, unstructured_dir: Path, max_chars: int = 6000):
        self.data_dir = unstructured_dir
        self.max_chars = max_chars
        self._docs: list[TextChunk] = []
        self._load()

    def _load(self) -> None:
        for txt in sorted(self.data_dir.glob("*_product_page.txt")):
            product_id = txt.stem.replace("_product_page", "")
            self._docs.append(
                TextChunk(
                    product_id=product_id,
                    filepath=txt,
                    content=txt.read_text(encoding="utf-8"),
                )
            )

    @property
    def product_ids(self) -> list[str]:
        return [d.product_id for d in self._docs]

    # ── Keyword scoring ───────────────────────────────────────────────────────

    def _score(self, doc: TextChunk, keywords: list[str]) -> float:
        text = doc.content.lower()
        return sum(text.count(kw.lower()) for kw in keywords)

    def _extract_keywords(self, question: str) -> list[str]:
        stopwords = {
            "what", "are", "the", "is", "a", "an", "of", "for",
            "do", "does", "about", "how", "well", "and", "that",
            "its", "it", "in", "to", "i", "want", "me",
        }
        tokens = re.findall(r"[a-z]+", question.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def search(self, question: str, top_k: int = 3) -> list[TextChunk]:
        """Return top_k most relevant product pages for the question."""
        keywords = self._extract_keywords(question)
        scored = [
            TextChunk(
                product_id=d.product_id,
                filepath=d.filepath,
                content=d.content,
                score=self._score(d, keywords),
            )
            for d in self._docs
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    # ── Section extractors ────────────────────────────────────────────────────

    def _extract_section(self, content: str, section_names: list[str]) -> str:
        """Extract a named section from a product page."""
        lines = content.splitlines()
        result: list[str] = []
        in_section = False
        for line in lines:
            if any(name.lower() in line.lower() for name in section_names):
                in_section = True
            elif in_section and line.startswith("##"):
                break
            if in_section:
                result.append(line)
        return "\n".join(result) if result else ""

    def get_reviews(self, product_id: str) -> str:
        for doc in self._docs:
            if doc.product_id == product_id:
                return self._extract_section(
                    doc.content, ["review", "customer review", "ratings"]
                )
        return ""

    def get_features(self, product_id: str) -> str:
        for doc in self._docs:
            if doc.product_id == product_id:
                return self._extract_section(
                    doc.content, ["feature", "specification", "key feature"]
                )
        return ""

    # ── Main retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = 3) -> str:
        """Search relevant product pages; extract targeted sections."""
        hits = self.search(question, top_k=top_k)
        q = question.lower()

        parts: list[str] = []
        for hit in hits:
            if hit.score == 0:
                continue

            header = f"=== {hit.product_id} ({hit.filepath.name}) [score={hit.score}] ==="
            parts.append(header)

            # Pull targeted sections based on question intent
            if any(w in q for w in ["review", "customer", "say", "rated", "rating", "ease", "cleaning"]):
                section = self.get_reviews(hit.product_id)
                parts.append(section if section else hit.content[:1500])
            elif any(w in q for w in ["feature", "specification", "spec", "key"]):
                section = self.get_features(hit.product_id)
                parts.append(section if section else hit.content[:1500])
            else:
                # Multi-source or general: include full page summary
                parts.append(hit.content[:2000])

        full = "\n\n".join(parts)
        return full[:self.max_chars] if full else "No relevant product pages found."
