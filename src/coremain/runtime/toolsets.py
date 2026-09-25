"""Role → tool set mapping. Roles are capabilities; tools outside a role's set are simply absent."""

from __future__ import annotations

from coremain.tools.base import Tool
from coremain.tools.control import (
    AskUser,
    RecordDecision,
    RecordExperiment,
    RecordHypothesis,
    SpawnSubtask,
    SubmitAnswer,
    SubmitPlan,
    SubmitResult,
    SubmitReview,
)
from coremain.tools.fs import (
    ApplyPatch,
    DeleteFile,
    EditFile,
    GlobFiles,
    ListDir,
    ReadFile,
    SearchText,
    WriteFile,
)
from coremain.tools.git_tools import GitBlame, GitCommit, GitDiff, GitLog, GitShow, GitStatus
from coremain.tools.knowledge import (
    FileOutline,
    FindReferences,
    FindSymbol,
    ImpactAnalysis,
    LoadSkill,
    MemoryPropose,
    MemorySearch,
    ProjectProfileTool,
    ReadSkillResource,
    RepoMap,
)
from coremain.tools.shell import RunCommand, RunTests

READ_ONLY: list[type[Tool]] = [
    ReadFile,
    ListDir,
    GlobFiles,
    SearchText,
    GitStatus,
    GitDiff,
    GitLog,
    GitShow,
    GitBlame,
    FindSymbol,
    FileOutline,
    FindReferences,
    ImpactAnalysis,
    RepoMap,
    ProjectProfileTool,
    MemorySearch,
    LoadSkill,
    ReadSkillResource,
]
WRITE: list[type[Tool]] = [WriteFile, EditFile, ApplyPatch, DeleteFile, RunCommand, RunTests, GitCommit]

ROLE_TOOLS: dict[str, list[type[Tool]]] = {
    "planner": [*READ_ONLY, SubmitPlan, RecordDecision, AskUser, SpawnSubtask],
    "implementer": [*READ_ONLY, *WRITE, MemoryPropose, RecordDecision, AskUser, SubmitResult],
    "corrector": [*READ_ONLY, *WRITE, MemoryPropose, AskUser, SubmitResult],
    "debugger": [
        *READ_ONLY,
        *WRITE,
        RecordHypothesis,
        RecordExperiment,
        MemoryPropose,
        AskUser,
        SubmitResult,
    ],
    "researcher": [*READ_ONLY, MemoryPropose, SpawnSubtask, SubmitAnswer],
    "reviewer": [*READ_ONLY, RunTests, SubmitReview],
}
MUTATING_ROLES = frozenset({"implementer", "corrector", "debugger"})


def tools_for(
    role: str, extra: list[Tool] | None = None, *, allow_spawn: bool = True, read_only_extra: bool = False
) -> list[Tool]:
    classes = ROLE_TOOLS.get(role, ROLE_TOOLS["researcher"])
    tools: list[Tool] = [cls() for cls in classes if allow_spawn or cls is not SpawnSubtask]
    names = {t.name for t in tools}
    for tool in extra or []:
        if tool.name in names:
            continue
        if (role not in MUTATING_ROLES or read_only_extra) and not tool.read_only:
            continue
        tools.append(tool)
        names.add(tool.name)
    return tools
