"""Evidence-rich final reports for tasks (Markdown)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coremain.domain.models import GateResult

if TYPE_CHECKING:
    from coremain.runtime.runner import TaskRun

STATUS_ICON = {
    "completed": "✓",
    "incomplete": "◐",
    "failed": "✗",
    "needs_input": "?",
    "blocked": "⏸",
    "cancelled": "⊘",
}


def render_report(
    run: TaskRun,
    gate: GateResult,
    *,
    apply_info: dict[str, Any] | None,
    final_status: str,
    conflict: dict[str, Any] | None = None,
) -> str:
    rt = run.rt
    lines: list[str] = []
    icon = STATUS_ICON.get(final_status, "•")
    level = f" · evidence level: {gate.level}" if run.changes_code or gate.level != "none" else ""
    lines.append(f"**{icon} Task {final_status}**{level}")
    answer = run.outputs.get("answer")
    result = run.outputs.get("result") or {}
    if answer:
        lines.append("")
        lines.append(answer.get("answer", "").strip())
        cites = answer.get("citations") or []
        if cites:
            refs = []
            for c in cites[:12]:
                span = (
                    f":{c['start_line']}" + (f"-{c['end_line']}" if c.get("end_line") else "")
                    if c.get("start_line")
                    else ""
                )
                refs.append(f"`{c['path']}{span}`")
            lines.append("")
            lines.append("Sources: " + ", ".join(refs))
    elif result.get("summary"):
        lines.append("")
        lines.append(result["summary"].strip())
    verification = run.outputs.get("verification") or {}
    if run.changes_code:
        lines.append("")
        files = [r for r in verification.get("results", []) if r["kind"] == "diff"]
        if files:
            lines.append(f"**Changes:** {files[-1]['summary']}")
        if apply_info:
            applied = len(apply_info.get("applied", [])) + len(apply_info.get("merged", []))
            lines.append(
                f"**Applied to your working tree:** {applied} file(s)"
                + (f" (3-way merged: {', '.join(apply_info['merged'])})" if apply_info.get("merged") else "")
            )
        elif run.workspace.isolated:
            lines.append(
                f"**Workspace:** changes are in `{run.workspace.path}` (branch `{run.workspace.branch or 'n/a'}`); "
                f"inspect with `core task diff {run.task.id}` and apply with `core task apply {run.task.id}`."
            )
        checks = [r for r in verification.get("results", []) if r["kind"] != "diff"]
        if checks:
            lines.append(
                "**Verification:** "
                + "; ".join(
                    f"{r['kind']}{(':' + r['name']) if r.get('name') and r['name'] != 'suite' else ''} "
                    f"{'✓' if r['status'] == 'pass' else r['status']} ({r['summary']})"
                    for r in checks
                )
            )
    reviews = rt.reviews.reviews(run.task.id)
    if reviews:
        last = [r for r in reviews if r["reviewer"] != "deterministic"][-3:] or reviews[-1:]
        open_findings = rt.reviews.open_findings(run.task.id)
        lines.append(
            "**Review:** "
            + "; ".join(
                f"{r['strategy']} by {r['reviewer']} → {r['verdict']}"
                + (f" ({r['independence']})" if r.get("independence") else "")
                for r in last
            )
        )
        for f in open_findings[:8]:
            loc = (
                f" `{f['file']}:{f['line']}`"
                if f.get("file") and f.get("line")
                else (f" `{f['file']}`" if f.get("file") else "")
            )
            lines.append(f"  - {f['severity']}: {f['title']}{loc}")
    if result.get("remaining_issues"):
        lines.append(
            "**Remaining issues (reported by implementer):** " + "; ".join(result["remaining_issues"][:8])
        )
    lines.append("")
    lines.append(f"**Evidence gate:** {'passed' if gate.passed else 'not satisfied'} — {gate.summary}")
    for note in gate.notes[:6]:
        lines.append(f"  - {note}")
    if conflict:
        lines.append("")
        lines.append(
            "**Apply conflict:** "
            + "; ".join(f"{c['path']} ({c['reason']})" for c in conflict.get("conflicts", [])[:10])
        )
    usage = run.budget.snapshot()
    lines.append("")
    lines.append(
        f"_{usage['model_calls']} model call(s), {usage['tokens']} tokens"
        + (f", ${usage['cost_usd']:.4f}" if usage["cost_usd"] else "")
        + f", task `{run.task.id}`_"
    )
    return "\n".join(lines)
