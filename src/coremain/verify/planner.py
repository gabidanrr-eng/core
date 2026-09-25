"""Verification planning: which evidence must exist before a task may complete."""

from __future__ import annotations

from typing import Any

from coremain.config.schema import VerificationConfig
from coremain.verify.evidence import EvidenceKind, Requirement

COMPILED = {"go", "rust", "java"}


def plan_requirements(
    *,
    changes_code: bool,
    kind: str,
    profile: dict[str, Any],
    contract: dict[str, Any],
    config: VerificationConfig,
) -> list[Requirement]:
    if not changes_code:
        reqs = [
            Requirement(
                EvidenceKind.WORKSPACE_UNCHANGED, description="no files were modified (read-only task)"
            )
        ]
        if kind in {"question", "research"}:
            reqs.append(
                Requirement(
                    EvidenceKind.CITATIONS,
                    description="answer cites files that exist and were actually read",
                    state_bound=False,
                )
            )
        if kind == "review":
            reqs.append(
                Requirement(EvidenceKind.REVIEW, description="structured review completed", state_bound=False)
            )
        return reqs
    commands = {**(profile.get("commands") or {}), **config.commands}
    reqs: list[Requirement] = [
        Requirement(EvidenceKind.DIFF, description="the workspace contains changes"),
        Requirement(EvidenceKind.SCOPE, description="no forbidden or protected paths were modified"),
        Requirement(EvidenceKind.NO_SECRETS, description="no secrets were introduced in the diff"),
    ]
    if commands.get("test"):
        reqs.append(Requirement(EvidenceKind.TESTS, "suite", f"test suite passes (`{commands['test']}`)"))
    else:
        reqs.append(Requirement(EvidenceKind.SYNTAX, description="changed files parse/compile"))
    ecosystems = set(profile.get("ecosystems") or [])
    if commands.get("build") and ecosystems & COMPILED:
        reqs.append(Requirement(EvidenceKind.BUILD, description=f"build succeeds (`{commands['build']}`)"))
    if commands.get("lint") and config.run_lint:
        reqs.append(
            Requirement(
                EvidenceKind.LINT,
                description=f"lint on changed files (`{commands['lint']}`)",
                required="lint" in (contract.get("required_checks") or []),
            )
        )
    if commands.get("typecheck") and config.run_typecheck:
        reqs.append(
            Requirement(
                EvidenceKind.TYPECHECK,
                description=f"type check (`{commands['typecheck']}`)",
                required="typecheck" in (contract.get("required_checks") or []),
            )
        )
    for item in contract.get("required_commands") or []:
        name = item.get("name") if isinstance(item, dict) else None
        cmd = item.get("command") if isinstance(item, dict) else str(item)
        reqs.append(Requirement(EvidenceKind.COMMAND, name or cmd, f"contract command passes (`{cmd}`)"))
    for path in contract.get("required_tests") or []:
        reqs.append(Requirement(EvidenceKind.TESTS, f"required:{path}", f"required test passes ({path})"))
    for check in contract.get("browser_checks") or []:
        name = check.get("name", "browser") if isinstance(check, dict) else str(check)
        reqs.append(Requirement(EvidenceKind.BROWSER, name, f"browser check passes ({name})"))
    if config.require_review:
        reqs.append(
            Requirement(
                EvidenceKind.REVIEW, description="review approved the current diff (no blocking findings)"
            )
        )
    return reqs


def requirements_to_dicts(reqs: list[Requirement]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in reqs]
