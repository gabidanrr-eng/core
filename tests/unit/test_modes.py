"""Request classification decides the workflow, so its edge cases are pinned here."""

from __future__ import annotations

import pytest

from coremain.routing.modes import classify


@pytest.mark.parametrize(
    ("text", "kind", "workflow"),
    [
        ("Review my uncommitted changes for security problems", "review", "review"),
        ("Please audit the last two commits", "review", "review"),
        ("Check this diff before I merge", "review", "review"),
        ("The login test is failing with a KeyError traceback", "bugfix", "debug"),
        ("Where is slugify defined and who calls it?", "question", "answer"),
        ("Add a /health endpoint with tests", "feature", "direct"),
        ("Add unit tests for the slugify module", "tests", "direct"),
        ("Increase test coverage of the parser", "tests", "direct"),
        ("Rename calc_total to compute_total everywhere", "refactor", "plan"),
        ("Document the CLI flags in the README", "docs", "direct"),
    ],
)
def test_classification(text: str, kind: str, workflow: str) -> None:
    cls = classify(text, repo_files=50)
    assert (cls.kind, cls.workflow) == (kind, workflow), cls.reasons


def test_review_request_that_asks_for_fixes_changes_code() -> None:
    cls = classify("Review my changes and fix any bugs you find", repo_files=50)
    assert cls.kind != "review" and cls.changes_code


def test_read_only_kinds_do_not_change_code() -> None:
    for text in ("Where is slugify defined?", "Review my uncommitted changes"):
        assert classify(text, repo_files=50).changes_code is False


def test_mode_override_wins_and_is_explained() -> None:
    cls = classify("Fix the failing test", repo_files=50, mode_override="plan")
    assert cls.mode == "plan" and cls.workflow == "plan"
    assert cls.reasons[0] == "mode overridden by user: plan"
