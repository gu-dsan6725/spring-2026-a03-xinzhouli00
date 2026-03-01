from __future__ import annotations

INTENT_RULES: dict[str, list[str]] = {
    "csv": [
        "revenue", "sales", "total", "units", "volume", "region",
        "category", "highest", "lowest", "average", "how much",
        "how many", "price", "december", "october", "november",
        "october", "quarter", "monthly", "daily", "trend", "selling", "sells", "well is it",
    ],
    "text": [
        "feature", "features", "review", "reviews", "customer",
        "specification", "spec", "description", "quality", "rated",
        "rating", "product page", "headphone", "air fryer", "what do",
        "key features", "ease", "cleaning", "recommend", "fitness",
    ],
}


def classify_query(question: str) -> list[str]:
    """
    Return all matching source tags for a question.

    Tags
    ----
    csv   → structured sales data (grep/awk/pandas on daily_sales.csv)
    text  → unstructured product pages (keyword / semantic search on *.txt)
    both  → when both tags fire, retriever combines sources

    Returns ['csv'] | ['text'] | ['csv', 'text']
    """
    ql = question.lower()
    tags = [tag for tag, kws in INTENT_RULES.items() if any(kw in ql for kw in kws)]
    # deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result if result else ["csv"]  
