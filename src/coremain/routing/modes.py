"""Task classification and execution-mode selection.

Deterministic, explainable heuristics decide how much process a request needs: complexity is
proportional to uncertainty, impact and scope. Users can always override the mode. The output
is concise operational metadata (never hidden reasoning), e.g. "security-sensitive change",
"ambiguous request", "large repository".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MODES = ("direct", "plan", "collaborative", "adversarial", "deep", "debug", "answer", "review", "recovery")
_Q_START = re.compile(
    r"^\s*(how|what|why|where|which|who|when|does|do|is|are|can|could|should|explain|describe|summari[sz]e|tell me|show me|list|find out|understand)\b",
    re.I,
)
_CHANGE = re.compile(
    r"\b(add|implement|create|build|write|fix|change|update|refactor|rename|remove|delete|migrate|upgrade|replace|"
    r"support|make|convert|port|integrate|optimi[sz]e|improve|move|extract|split|introduce|enable|disable|set up|setup)\b",
    re.I,
)
_DEBUG = re.compile(
    r"\b(fail(s|ing|ed|ure)?|error|exception|traceback|crash(es|ing)?|bug|broken|regression|flaky|hang(s|ing)?|"
    r"doesn'?t work|not working|investigate why|root cause|stack ?trace|segfault|timeout)\b",
    re.I,
)
_REVIEW = re.compile(
    r"\b(review|audit|critique|assess|check)\b.*\b(diffs?|changes?|prs?|pull requests?|code|implementations?|commits?|"
    r"branch(es)?|patch(es)?|edits?|modifications?)\b",
    re.I,
)
_REFACTOR = re.compile(
    r"\b(refactor|restructure|reorgani[sz]e|clean ?up|decouple|extract|rename|modulari[sz]e)\b", re.I
)
_ARCH = re.compile(
    r"\b(architecture|design|approach(es)?|trade-?offs?|compare|alternatives?|options|strategy|rfc)\b", re.I
)
_SECURITY = re.compile(
    r"\b(auth(entication|orization)?|login|password|token|secret|crypto|encrypt|jwt|oauth|session|permission|"
    r"rbac|csrf|xss|sql injection|injection|sanitiz|payment|billing|pii|security|vulnerab|cve|certificate|tls|ssrf)\b",
    re.I,
)
_UI = re.compile(
    r"\b(ui|frontend|front-end|page|button|form|css|layout|responsive|browser|click|render|component|screen|modal|dom)\b",
    re.I,
)
_DOCS = re.compile(r"\b(readme|docs?|documentation|docstring|changelog|comment)\b", re.I)
_TESTS = re.compile(r"\b(tests?|coverage|unit test|integration test|e2e)\b", re.I)
# Tests are the deliverable ("add tests for X", "increase coverage"), not a side requirement.
_TESTS_TASK = re.compile(
    r"\b(add|write|create|increase|improve|expand)\s+(?:(?:some|more|missing|the|a|new|better|unit|integration|e2e|"
    r"end-to-end|regression|property-based)\s+)*(tests?|test cases?|test coverage|coverage|specs?)\b|^\s*tests?\s+for\b",
    re.I,
)
_DEEP = re.compile(
    r"\b(thorough(ly)?|carefully|deep|comprehensive|production[- ]ready|robust|hardened?|critical)\b", re.I
)
_RESUME = re.compile(
    r"\b(continue|resume|pick up|finish)\b.*\b(task|work|yesterday|previous|unfinished|where)\b", re.I
)
_RESEARCH = re.compile(
    r"\b(research|look up|find documentation|latest version|what is the best|osint|gather information)\b",
    re.I,
)
_MULTI = re.compile(r"\b(and|also|then|plus|as well as)\b", re.I)


@dataclass
class Classification:
    kind: str
    mode: str
    workflow: str
    risk: str
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    review_strategies: list[str] = field(default_factory=list)
    changes_code: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


WORKFLOW_FOR_MODE = {
    "direct": "direct",
    "plan": "plan",
    "collaborative": "collaborative",
    "adversarial": "plan",
    "deep": "plan",
    "debug": "debug",
    "answer": "answer",
    "review": "review",
    "recovery": "plan",
}


def classify(
    text: str,
    *,
    repo_files: int = 0,
    mode_override: str | None = None,
    models_available: int = 1,
    has_tests: bool = True,
) -> Classification:
    t = text.strip()
    reasons: list[str] = []
    signals: dict[str, Any] = {}
    is_question = bool(_Q_START.search(t)) or (t.endswith("?") and not _CHANGE.search(t))
    changes = bool(_CHANGE.search(t))
    debug = bool(_DEBUG.search(t))
    review = bool(_REVIEW.search(t)) and not changes
    security = bool(_SECURITY.search(t))
    ui = bool(_UI.search(t))
    arch = bool(_ARCH.search(t))
    deep = bool(_DEEP.search(t))
    words = len(t.split())
    ambiguous = words < 6 and not is_question and not debug
    multi = len(_MULTI.findall(t)) >= 3 or words > 120
    signals.update(
        {
            "question": is_question,
            "changes": changes,
            "debug": debug,
            "review": review,
            "security": security,
            "ui": ui,
            "architecture": arch,
            "deep": deep,
            "ambiguous": ambiguous,
            "multi_part": multi,
            "repo_files": repo_files,
        }
    )
    if _RESUME.search(t):
        kind = "resume"
    elif review:
        kind = "review"
    elif is_question and not changes and not (debug and "fix" in t.lower()):
        kind = "research" if _RESEARCH.search(t) else "question"
    elif debug:
        kind = "bugfix"
    elif _REFACTOR.search(t):
        kind = "refactor"
    elif _DOCS.search(t) and not _CODE_HINT.search(t):
        kind = "docs"
    elif _TESTS_TASK.search(t):
        kind = "tests"
    elif arch and not changes:
        kind = "architecture"
    else:
        kind = "feature"
    risk = "normal"
    if security:
        risk = "high"
        reasons.append("security-sensitive request")
    elif kind in {"docs", "question", "research"}:
        risk = "low"
    if repo_files > 3000:
        reasons.append("large repository")
    if ui:
        reasons.append("UI/browser behaviour involved")
    if ambiguous:
        reasons.append("short/ambiguous request")
    if multi:
        reasons.append("multi-part request")
    review_strategies = ["correctness"]
    if kind in {"question", "research"}:
        mode = "answer"
        reasons.insert(0, "read-only question")
    elif kind == "review":
        mode = "review"
        reasons.insert(0, "review of existing changes requested")
        review_strategies = ["correctness", "security"] if security else ["correctness"]
    elif kind == "resume":
        mode = "recovery"
        reasons.insert(0, "continuation of earlier work")
    elif kind == "bugfix":
        mode = "debug"
        reasons.insert(0, "failure investigation: reproduce → hypothesize → fix → regression test")
    elif kind == "architecture":
        mode = "collaborative" if models_available > 1 else "plan"
        reasons.insert(
            0, "architectural comparison" + (" with independent proposals" if models_available > 1 else "")
        )
        review_strategies = ["correctness", "architecture"]
    elif deep:
        mode = "deep"
        reasons.insert(0, "thorough treatment requested")
        review_strategies = ["correctness", "tests", "architecture"] + (["security"] if security else [])
    elif security:
        mode = "adversarial"
        reasons.insert(0, "independent security review triggered")
        review_strategies = ["correctness", "security", "adversarial"]
    elif kind == "refactor" or multi or ambiguous or repo_files > 3000 or words > 60:
        mode = "plan"
        reasons.insert(
            0, "planning first (" + ("refactor" if kind == "refactor" else "scope/uncertainty") + ")"
        )
        if kind == "refactor":
            review_strategies = ["correctness", "tests"]
    else:
        mode = "direct"
        reasons.insert(0, "small, well-scoped change")
    if mode_override:
        if mode_override not in MODES:
            raise ValueError(f"unknown mode '{mode_override}'; choose from {', '.join(MODES)}")
        reasons.insert(0, f"mode overridden by user: {mode_override}")
        mode = mode_override
        if mode == "adversarial" and "adversarial" not in review_strategies:
            review_strategies = ["correctness", "adversarial"]
    if not has_tests and kind not in {"question", "research", "review", "docs"}:
        reasons.append("no test suite detected: verification evidence will be weaker")
    changes_code = mode not in {"answer", "review"}
    return Classification(
        kind, mode, WORKFLOW_FOR_MODE[mode], risk, reasons, signals, review_strategies, changes_code
    )


_CODE_HINT = re.compile(r"\b(function|class|method|endpoint|api|bug|test|module|script)\b", re.I)
