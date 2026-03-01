from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import List

from .bash_tools import BashToolRunner, CommandPlan, build_search_cmd
from .llm import LLMClient
from .utils import QAResult, dedent, normalize_whitespace


TEST_QUESTIONS: List[str] = [
    "What Python dependencies does this project use?",
    "What is the main entry point file for the registry service?",
    "What programming languages and file types are used in this repository? (e.g., Python, TypeScript, YAML, JSON, Dockerfile, etc.)",
    "How does the authentication flow work, from token validation to user authorization?",
    "What are all the API endpoints available in the registry service and what scopes do they require?",
    "How would you add support for a new OAuth provider (e.g., Okta) to the authentication system? What files would need to be modified and what interfaces must be implemented?",
]


# ── 1. Query classifier ────────────────────────────────────────────────────────

INTENT_RULES: dict[str, list[str]] = {
    "dependencies": [
        "dependency", "dependencies", "requirements", "package",
        "pyproject", "pip", "npm", "install", "library", "libraries",
    ],
    "entry": [
        "entry point", "entrypoint", "main file", "start", "launch",
        "run", "server", "how to run", "how to start",
    ],
    "languages": [
        "language", "file type", "file types", "typescript", "python",
        "yaml", "dockerfile", "technology", "stack", "programming",
    ],
    "auth_flow": [
        "auth", "authentication", "authorization", "token", "oauth",
        "jwt", "login", "scope", "permission", "credential",
    ],
    "api_endpoints": [
        "api", "endpoint", "route", "rest", "swagger", "openapi",
        "get ", "post ", "put ", "delete ", "patch ", "http",
    ],
    "oauth_provider": [
        "add", "implement", "integrate", "new provider", "okta",
        "how would", "what files", "modify", "extend", "support for",
    ],
}


def classify_query(question: str) -> list[str]:
    """
    Return ALL matching intent tags for a question (multi-label).
    """
    ql = question.lower()
    tags = [
        tag
        for tag, keywords in INTENT_RULES.items()
        if any(kw in ql for kw in keywords)
    ]
    return tags if tags else ["general"]


# ── 2. Per-intent command sets ─────────────────────────────────────────────────

def _commands_for(tag: str, use_rg: bool) -> List[str]:
    """Return the bash commands for a single intent tag."""

    if tag == "dependencies":
        return [
            "cat pyproject.toml",
            "find . -maxdepth 4 -name 'package.json' -not -path '*/node_modules/*' -type f",
            "cat package.json 2>/dev/null || true",
            "cat cli/package.json 2>/dev/null || true",
            "cat frontend/package.json 2>/dev/null || true",
        ]

    if tag == "entry":
        return [
            build_search_cmd(
                "app = FastAPI|if __name__",
                ["*.py"], use_rg,
                roots=["registry", "auth_server"],
                limit=60,
            ),
            build_search_cmd(
                "uvicorn",
                ["*.py", "*.toml", "Makefile", "*.sh"], use_rg,
                roots=["."],
                limit=40,
            ),
            "find . -name 'main.py' -not -path '*/node_modules/*'",
            "cat registry/main.py 2>/dev/null || true",
        ]

    if tag == "languages":
        return [
            (
                "find . -type f -not -path '*/.git/*' -not -path '*/node_modules/*'"
                " | sed 's/.*\\.//' | sort | uniq -c | sort -rn | head -30"
            ),
            "find . -name 'Dockerfile*' -not -path '*/node_modules/*'",
            "find . -name '*.yaml' -o -name '*.yml' | grep -v node_modules | head -20",
        ]

    if tag == "auth_flow":
        return [
            build_search_cmd(
                "token|jwt|bearer|scope|verify_token|oauth",
                ["*.py"], use_rg,
                roots=["registry", "auth_server"],
                limit=200,
            ),
            build_search_cmd(
                "auth|token|scope",
                ["*.md"], use_rg,
                roots=["docs"],
                limit=200,
            ),
            "find auth_server -type f -name '*.py' 2>/dev/null || true",
            "find docs -type f 2>/dev/null || true",
        ]

    if tag == "api_endpoints":
        return [
            build_search_cmd(
                r"@router\.(get|post|put|patch|delete)",
                ["*.py"], use_rg,
                roots=["registry"],
                limit=200,
            ),
            build_search_cmd(
                "APIRouter|include_router",
                ["*.py"], use_rg,
                roots=["registry"],
                limit=100,
            ),
            build_search_cmd(
                "scope|Security|Depends",
                ["*.py"], use_rg,
                roots=["registry"],
                limit=100,
            ),
            "find registry -name '*.py' -type f",
        ]

    if tag == "oauth_provider":
        return [
            build_search_cmd(
                "oauth|provider|OAuthProvider|get_provider",
                ["*.py"], use_rg,
                roots=["auth_server", "registry"],
                limit=200,
            ),
            build_search_cmd(
                "oauth|provider",
                ["*.md"], use_rg,
                roots=["docs"],
                limit=200,
            ),
            "find auth_server -type f 2>/dev/null || true",
            "find docs -type f 2>/dev/null || true",
        ]

    # general fallback
    return [
        "tree -L 2 2>/dev/null || find . -maxdepth 2 -not -path '*/.git/*'",
        build_search_cmd(
            "registry",
            ["*.py", "*.md"], use_rg,
            roots=["registry", "docs"],
            limit=100,
        ),
    ]


def plan_commands(tags: list[str]) -> CommandPlan:
    use_rg = shutil.which("rg") is not None
    seen: set[str] = set()
    merged: List[str] = []

    for tag in tags:
        for cmd in _commands_for(tag, use_rg):
            if cmd not in seen:
                seen.add(cmd)
                merged.append(cmd)

    return CommandPlan(commands=merged)


# ── 3. Prompt builder ──────────────────────────────────────────────────────────

def build_prompt(question: str, context: str) -> str:
    return dedent(f"""
        You are answering questions about the mcp-gateway-registry codebase.
        Use ONLY the context below. Cite file paths and line numbers where available.
        Be precise and concise.

        Question:
        {question}

        Context:
        {context}
    """)


# ── 4. Single-question entry point ─────────────────────────────────────────────

def answer_question(
    repo_root: Path,
    question: str,
    max_context_chars: int = 6000,
) -> QAResult:
    """Classify → plan → execute bash commands → call LLM → return QAResult."""
    tags = classify_query(question)
    plan = plan_commands(tags)
    runner = BashToolRunner(repo_root, max_chars=max_context_chars)

    outputs: List[str] = []
    commands_run: List[str] = []

    for cmd in plan.commands:
        result = runner.run(cmd)
        commands_run.append(cmd)
        if result.output:
            outputs.append(f"$ {cmd}\n{result.output}")

    context = normalize_whitespace("\n\n".join(outputs))
    prompt = build_prompt(question, context)

    llm = LLMClient()
    answer = llm.generate(prompt)

    if "LLM unavailable" in answer:
        answer = (
            "LLM not configured — add GROQ_API_KEY (free) to .env.\n"
            "Context captured above can be used for manual answering."
        )

    return QAResult(
        question=question,
        answer=answer,
        context=context,
        commands=commands_run,
    )


# ── 5. Batch runner ────────────────────────────────────────────────────────────

def run_all_questions(repo_root: Path, output_path: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Codebase not found at {repo_root}.\n"
            "Run: git clone https://github.com/agentic-community/mcp-gateway-registry"
        )

    throttle = float(os.getenv("LLM_THROTTLE_SECONDS", "0"))
    sections: List[str] = []

    for idx, question in enumerate(TEST_QUESTIONS, start=1):
        tags = classify_query(question)
        print(f"[{idx}/{len(TEST_QUESTIONS)}] {tags} | {question[:60]}...")
        result = answer_question(repo_root, question)
        sections.append(f"# Question {idx}\n{result.format_for_file()}")
        if throttle > 0 and idx < len(TEST_QUESTIONS):
            time.sleep(throttle)

    output_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    print(f"\n  Written → {output_path}")


# ── CLI shortcut ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _root = Path(os.getenv("MCP_REGISTRY_PATH", "./mcp-gateway-registry"))
    _out = Path("part1_results.txt")
    run_all_questions(_root, _out)