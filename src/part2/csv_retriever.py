from __future__ import annotations

import subprocess
from pathlib import Path

from part1.bash_tools import BashToolRunner

class CSVRetriever:
    """
    Retrieves context from daily_sales.csv using bash tools (grep, awk, cut)
    and pandas for aggregation.
    """

    def __init__(self, csv_path, max_chars=6000):
        self.csv_path = csv_path
        self.max_chars = max_chars
        self._runner = BashToolRunner(cwd=csv_path.parent, max_chars=max_chars)

    def _run(self, cmd):
        return self._runner.run(cmd).output

    # ── Bash-based retrieval ──────────────────────────────────────────────────

    def get_header(self) -> str:
        return self._run(f"head -1 {self.csv_path.name}")

    def grep_category(self, category: str) -> str:
        return self._run(
            f"grep -i '{category}' {self.csv_path.name} | head -50"
        )

    def grep_region(self, region: str) -> str:
        return self._run(
            f"grep -i '{region}' {self.csv_path.name} | head -50"
        )

    def grep_month(self, month: str) -> str:
        """month like '2024-12' """
        return self._run(
            f"grep '{month}' {self.csv_path.name} | head -100"
        )

    def awk_total_revenue_by_category(self, category: str) -> str:
        """Sum total_revenue (col 7) for rows matching category (col 4)."""
        return self._run(
            f"awk -F',' 'tolower($4) ~ /{category.lower()}/ {{sum += $7}} "
            f"END {{print \"Total revenue for {category}: $\" sum}}' {self.csv_path.name}"
        )

    def awk_units_by_region(self) -> str:
        """Sum units_sold (col 6) grouped by region (col 8)."""
        return self._run(
            f"awk -F',' 'NR>1 {{units[$8] += $6}} "
            f"END {{for (r in units) print r, units[r]}}' {self.csv_path.name}"
            f" | sort -k2 -rn"
        )

    # ── Pandas-based aggregation (richer, used for multi-source questions) ───

    def pandas_summary(self, question: str) -> str:
        """Run pandas aggregations relevant to the question, return as text."""
        try:
            import pandas as pd
        except ImportError:
            return "pandas not available — install with: pip install pandas"

        df = pd.read_csv(self.csv_path)
        df["date"] = pd.to_datetime(df["date"])
        lines: list[str] = []

        q = question.lower()

        # Revenue by category
        if any(w in q for w in ["revenue", "sales", "category"]):
            rev = (
                df.groupby("category")["total_revenue"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            rev["total_revenue"] = rev["total_revenue"].map("${:,.2f}".format)
            lines.append("=== Total Revenue by Category ===")
            lines.append(rev.to_string(index=False))

        # Units by region
        if any(w in q for w in ["region", "volume", "units"]):
            reg = (
                df.groupby("region")["units_sold"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            lines.append("\n=== Units Sold by Region ===")
            lines.append(reg.to_string(index=False))

        # December filter
        if "december" in q or "dec" in q:
            dec = df[df["date"].dt.month == 12]
            dec_rev = (
                dec.groupby("category")["total_revenue"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            dec_rev["total_revenue"] = dec_rev["total_revenue"].map("${:,.2f}".format)
            lines.append("\n=== December 2024 Revenue by Category ===")
            lines.append(dec_rev.to_string(index=False))

        # West region filter
        if "west" in q:
            west = df[df["region"].str.lower() == "west"]
            west_prod = (
                west.groupby("product_name")["units_sold"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            lines.append("\n=== Top 10 Products in West Region (units sold) ===")
            lines.append(west_prod.to_string(index=False))

        # Top selling products (always include for multi-source)
        if any(w in q for w in ["best", "top", "selling", "recommend", "fitness"]):
            top = (
                df.groupby("product_name")["units_sold"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            lines.append("\n=== Top 15 Products by Units Sold ===")
            lines.append(top.to_string(index=False))

        result = "\n".join(lines) if lines else df.describe().to_string()
        return result[:self.max_chars]

    def retrieve(self, question: str) -> str:
        """Main entry point — combine bash + pandas context for a question."""
        parts: list[str] = []

        parts.append(f"$ head -1 {self.csv_path.name}\n{self.get_header()}")

        # Always include pandas aggregations (they're the most useful)
        pandas_ctx = self.pandas_summary(question)
        if pandas_ctx:
            parts.append(f"[pandas aggregations]\n{pandas_ctx}")

        q = question.lower()

        # Bash supplements
        if "december" in q or "dec" in q:
            parts.append(
                f"$ grep '2024-12' ...\n{self.grep_month('2024-12')}"
            )
        if "electronics" in q:
            parts.append(
                f"$ awk total_revenue electronics\n{self.awk_total_revenue_by_category('Electronics')}"
            )
        if "region" in q or "volume" in q:
            parts.append(
                f"$ awk units_by_region\n{self.awk_units_by_region()}"
            )

        full = "\n\n".join(parts)
        return full[:self.max_chars]
