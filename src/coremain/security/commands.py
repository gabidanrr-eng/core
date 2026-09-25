"""Static risk analysis for shell commands.

This is not a sandbox. It classifies what a command is likely to do so the policy engine can
allow, ask or deny before execution, and so audit records capture the reasoning.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from coremain.security.paths import sensitive_reason


class CommandRisk(StrEnum):
    READ_ONLY = "read_only"
    GIT_READ = "git_read"
    TEST = "test"
    BUILD = "build"
    LOCAL_EXEC = "local_exec"
    LOCAL_MUTATION = "local_mutation"
    GIT_WRITE = "git_write"
    PACKAGE_INSTALL = "package_install"
    NETWORK = "network"
    EXTERNAL = "external_side_effect"
    DESTRUCTIVE = "destructive"
    PRIVILEGED = "privileged"
    SECRET_ACCESS = "secret_access"  # noqa: S105 - risk label, not a credential
    UNKNOWN = "unknown"


SEVERITY = {
    CommandRisk.READ_ONLY: 0,
    CommandRisk.GIT_READ: 0,
    CommandRisk.TEST: 1,
    CommandRisk.BUILD: 1,
    CommandRisk.LOCAL_EXEC: 2,
    CommandRisk.LOCAL_MUTATION: 2,
    CommandRisk.GIT_WRITE: 2,
    CommandRisk.UNKNOWN: 3,
    CommandRisk.PACKAGE_INSTALL: 3,
    CommandRisk.NETWORK: 3,
    CommandRisk.DESTRUCTIVE: 4,
    CommandRisk.EXTERNAL: 5,
    CommandRisk.PRIVILEGED: 5,
    CommandRisk.SECRET_ACCESS: 5,
}


@dataclass
class CommandAnalysis:
    risks: set[CommandRisk] = field(default_factory=set)
    programs: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    parse_ok: bool = True

    @property
    def max_risk(self) -> CommandRisk:
        if not self.risks:
            return CommandRisk.UNKNOWN
        return max(self.risks, key=lambda r: SEVERITY[r])

    def add(self, risk: CommandRisk, reason: str) -> None:
        self.risks.add(risk)
        if reason not in self.reasons:
            self.reasons.append(reason)


READ_ONLY = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "fd",
    "tree",
    "pwd",
    "echo",
    "printf",
    "which",
    "whereis",
    "type",
    "file",
    "stat",
    "du",
    "df",
    "diff",
    "cmp",
    "sort",
    "uniq",
    "cut",
    "tr",
    "jq",
    "yq",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "date",
    "uname",
    "whoami",
    "id",
    "true",
    "false",
    "test",
    "[",
    "sleep",
    "seq",
    "nl",
    "column",
    "od",
    "xxd",
    "hexdump",
    "strings",
    "md5sum",
    "sha1sum",
    "sha256sum",
    "less",
    "more",
    "comm",
    "paste",
    "tac",
    "rev",
    "fold",
    "expand",
    "hostname",
    "nproc",
    "free",
    "uptime",
    "ps",
    "lsof",
    "tokei",
    "cloc",
    "awk",
    "gawk",
}
TEST_PROGRAMS = {
    "pytest",
    "tox",
    "nox",
    "jest",
    "vitest",
    "mocha",
    "ava",
    "rspec",
    "phpunit",
    "ctest",
    "bats",
}
BUILD_PROGRAMS = {
    "ruff",
    "black",
    "isort",
    "flake8",
    "pylint",
    "mypy",
    "pyright",
    "eslint",
    "prettier",
    "tsc",
    "gcc",
    "g++",
    "cc",
    "clang",
    "clang++",
    "javac",
    "rustc",
    "gofmt",
    "shellcheck",
    "hadolint",
    "biome",
    "stylelint",
}
INTERPRETERS = {
    "python",
    "python3",
    "node",
    "bun",
    "deno",
    "ruby",
    "perl",
    "php",
    "bash",
    "sh",
    "zsh",
    "lua",
    "Rscript",
}
MUTATING = {
    "mv",
    "cp",
    "mkdir",
    "touch",
    "ln",
    "rmdir",
    "install",
    "patch",
    "tee",
    "chmod",
    "unzip",
    "tar",
    "gzip",
    "gunzip",
    "zip",
}
NETWORK_PROGRAMS = {
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "nc",
    "ncat",
    "netcat",
    "telnet",
    "ftp",
    "http",
    "https",
    "dig",
    "nslookup",
    "ping",
    "traceroute",
    "aria2c",
    "socat",
    "rsync",
}
EXTERNAL_PROGRAMS = {
    "aws",
    "gcloud",
    "az",
    "kubectl",
    "helm",
    "terraform",
    "pulumi",
    "vercel",
    "netlify",
    "fly",
    "flyctl",
    "heroku",
    "twine",
}
PRIVILEGED_PROGRAMS = {
    "sudo",
    "su",
    "doas",
    "pkexec",
    "chroot",
    "mount",
    "umount",
    "systemctl",
    "service",
    "iptables",
    "useradd",
    "userdel",
    "usermod",
    "passwd",
    "visudo",
    "crontab",
    "launchctl",
}
DESTRUCTIVE_PROGRAMS = {
    "dd",
    "mkfs",
    "shred",
    "wipefs",
    "fdisk",
    "parted",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "killall",
    "pkill",
}
SECRET_PROGRAMS = {"printenv", "security", "keyctl", "secret-tool", "history"}
PACKAGE_MANAGERS = {
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "bun",
    "pip",
    "pip3",
    "uv",
    "poetry",
    "pipenv",
    "cargo",
    "go",
    "gem",
    "bundle",
    "composer",
    "mvn",
    "gradle",
    "dotnet",
    "apt",
    "apt-get",
    "brew",
    "conda",
    "mamba",
    "make",
}
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "|&", ";;", "\n"}
_REDIRECTS = {">", ">>", "<", "<<", "<<<", ">&", "&>", "2>", "2>>", ">|"}
_GIT_READ_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "blame",
    "branch",
    "rev-parse",
    "ls-files",
    "grep",
    "describe",
    "shortlog",
    "reflog",
    "cat-file",
    "ls-tree",
    "config",
    "remote",
    "tag",
    "stash",
    "worktree",
    "merge-base",
    "name-rev",
    "rev-list",
    "whatchanged",
}
_GIT_REMOTE_SUBCOMMANDS = {"push", "send-email", "request-pull"}
_GIT_NETWORK_SUBCOMMANDS = {"fetch", "pull", "clone", "ls-remote", "submodule"}


def _segments(command: str) -> tuple[list[list[str]], bool]:
    text = command.replace("`", " ( ").replace("$(", " ( ")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return [command.split()], False
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return [s for s in segments if s], True


def _outside(workspace: Path | None, arg: str) -> bool:
    if workspace is None or arg.startswith("-"):
        return False
    ws = workspace.resolve()
    if arg in {"/", "~", "*", "/*", "~/", "$HOME", ".."} or arg.startswith(("~", "$HOME")):
        return True
    try:
        target = (ws / os.path.expanduser(arg)).resolve()
        return not target.is_relative_to(ws)
    except (OSError, ValueError):
        return True


def _classify_segment(seg: list[str], analysis: CommandAnalysis, workspace: Path | None, depth: int) -> None:
    words = list(seg)
    while words and _ASSIGNMENT.match(words[0]):
        words.pop(0)
    if not words:
        return
    for i, w in enumerate(words):
        if (w in _REDIRECTS or w.endswith(">")) and i + 1 < len(words):
            target = words[i + 1]
            if target.startswith("/dev/") and target not in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
                analysis.add(CommandRisk.DESTRUCTIVE, f"writes to device {target}")
            elif ">" in w:
                if _outside(workspace, target):
                    analysis.add(
                        CommandRisk.DESTRUCTIVE, f"redirects output outside the workspace ({target})"
                    )
                elif target != "/dev/null":
                    analysis.add(CommandRisk.LOCAL_MUTATION, f"redirects output to {target}")
    argv = [
        w for i, w in enumerate(words) if w not in _REDIRECTS and not (i > 0 and words[i - 1] in _REDIRECTS)
    ]
    if not argv:
        return
    prog_path = argv[0]
    prog = os.path.basename(prog_path)
    analysis.programs.append(prog)
    args = argv[1:]
    for a in args:
        reason = sensitive_reason(a) if not a.startswith("-") else None
        if reason and prog not in {"echo", "printf", "ls", "touch", "test", "["}:
            analysis.add(CommandRisk.SECRET_ACCESS, f"argument '{a}' {reason}")
    if prog in PRIVILEGED_PROGRAMS:
        analysis.add(CommandRisk.PRIVILEGED, f"'{prog}' requires elevated privileges")
        if prog in {"sudo", "doas"} and args:
            _classify_segment(args, analysis, workspace, depth + 1)
        return
    if prog in SECRET_PROGRAMS:
        analysis.add(CommandRisk.SECRET_ACCESS, f"'{prog}' can expose secrets")
        return
    if prog == "env":
        rest = [a for a in args if not _ASSIGNMENT.match(a) and not a.startswith("-")]
        if not rest:
            analysis.add(CommandRisk.SECRET_ACCESS, "'env' without a command prints the environment")
        else:
            _classify_segment(rest, analysis, workspace, depth + 1)
        return
    if prog in {"xargs", "nohup", "time", "timeout", "nice", "stdbuf", "exec", "command"}:
        rest = [a for a in args if not a.startswith("-")]
        if prog == "timeout" and rest:
            rest = rest[1:]
        if rest:
            _classify_segment(rest, analysis, workspace, depth + 1)
        else:
            analysis.add(CommandRisk.READ_ONLY, f"'{prog}' wrapper")
        return
    if prog in DESTRUCTIVE_PROGRAMS or prog.startswith("mkfs"):
        analysis.add(CommandRisk.DESTRUCTIVE, f"'{prog}' is destructive")
        return
    if prog == "kill":
        if "-1" in args:
            analysis.add(CommandRisk.DESTRUCTIVE, "kills all processes")
        else:
            analysis.add(CommandRisk.LOCAL_MUTATION, "sends a signal to a process")
        return
    if prog == "rm":
        flags = "".join(a.lstrip("-") for a in args if a.startswith("-"))
        targets = [a for a in args if not a.startswith("-")]
        if not targets or any(_outside(workspace, t) for t in targets):
            analysis.add(CommandRisk.DESTRUCTIVE, "rm targets paths outside the workspace")
        elif "r" in flags.lower() and any(t in {".", "*", "./*", "./"} for t in targets):
            analysis.add(CommandRisk.DESTRUCTIVE, "rm -r on the whole workspace")
        else:
            analysis.add(CommandRisk.LOCAL_MUTATION, "removes files inside the workspace")
        return
    if prog in {"chmod", "chown"} and any(_outside(workspace, a) for a in args[1:] if not a.startswith("-")):
        analysis.add(CommandRisk.DESTRUCTIVE, f"{prog} outside the workspace")
        return
    if prog == "find":
        if any(a in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for a in args):
            analysis.add(CommandRisk.LOCAL_MUTATION, "find with -delete/-exec")
            if "-delete" in args and any(_outside(workspace, a) for a in args if not a.startswith("-")):
                analysis.add(CommandRisk.DESTRUCTIVE, "find -delete outside the workspace")
        else:
            analysis.add(CommandRisk.READ_ONLY, "find (read-only)")
        return
    if prog == "sed":
        if any(a == "-i" or a.startswith("-i") or a == "--in-place" for a in args):
            analysis.add(CommandRisk.LOCAL_MUTATION, "sed -i edits files in place")
        else:
            analysis.add(CommandRisk.READ_ONLY, "sed (stream)")
        return
    if prog == "git":
        _classify_git(args, analysis)
        return
    if prog == "gh":
        sub = " ".join(a for a in args[:2] if not a.startswith("-"))
        if sub.startswith(
            (
                "pr create",
                "pr merge",
                "pr close",
                "release",
                "repo create",
                "repo delete",
                "issue create",
                "issue close",
                "workflow run",
                "secret",
                "api",
            )
        ):
            analysis.add(CommandRisk.EXTERNAL, f"'gh {sub}' changes remote state")
        elif sub.startswith("auth"):
            analysis.add(CommandRisk.SECRET_ACCESS, "'gh auth' can reveal credentials")
        else:
            analysis.add(CommandRisk.NETWORK, f"'gh {sub}' queries GitHub")
        return
    if prog in {"docker", "podman"}:
        sub = args[0] if args else ""
        if sub in {"push", "login"}:
            analysis.add(CommandRisk.EXTERNAL, f"'{prog} {sub}' affects a remote registry")
        elif sub in {"rm", "rmi", "system", "volume", "prune"}:
            analysis.add(CommandRisk.DESTRUCTIVE, f"'{prog} {sub}' removes resources")
        else:
            analysis.add(CommandRisk.UNKNOWN, f"'{prog} {sub}' runs containers")
        return
    if prog in EXTERNAL_PROGRAMS:
        analysis.add(CommandRisk.EXTERNAL, f"'{prog}' operates on external infrastructure")
        return
    if prog in NETWORK_PROGRAMS:
        analysis.add(CommandRisk.NETWORK, f"'{prog}' performs network access")
        return
    if prog in PACKAGE_MANAGERS:
        _classify_package_manager(prog, args, analysis, workspace)
        return
    if prog in TEST_PROGRAMS:
        analysis.add(CommandRisk.TEST, f"'{prog}' runs tests")
        return
    if prog in BUILD_PROGRAMS:
        analysis.add(CommandRisk.BUILD, f"'{prog}' is a build/lint tool")
        return
    if prog in INTERPRETERS or re.match(r"^python3(\.\d+)?$", prog):
        _classify_interpreter(prog, args, analysis, workspace, depth)
        return
    if prog in READ_ONLY:
        analysis.add(CommandRisk.READ_ONLY, f"'{prog}' is read-only")
        return
    if prog in MUTATING:
        if prog not in {"tar", "unzip"} and any(
            _outside(workspace, a) for a in args if not a.startswith("-")
        ):
            analysis.add(CommandRisk.DESTRUCTIVE, f"'{prog}' writes outside the workspace")
        else:
            analysis.add(CommandRisk.LOCAL_MUTATION, f"'{prog}' modifies files")
        return
    if prog_path.startswith(("./", "../")) or "/" in prog_path:
        if workspace is not None and not _outside(workspace, prog_path):
            analysis.add(CommandRisk.LOCAL_EXEC, f"runs workspace program '{prog_path}'")
        else:
            analysis.add(CommandRisk.UNKNOWN, f"runs program outside the workspace '{prog_path}'")
        return
    analysis.add(CommandRisk.UNKNOWN, f"unrecognized program '{prog}'")


def _classify_git(args: list[str], analysis: CommandAnalysis) -> None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in {"-C", "-c", "--git-dir", "--work-tree"} else 1
    rest = args[i:]
    sub = rest[0] if rest else ""
    tail = rest[1:]
    flags = set(tail)
    if sub in _GIT_REMOTE_SUBCOMMANDS:
        if flags & {"--force", "-f", "--force-with-lease", "--mirror", "--delete", "-d"}:
            analysis.add(CommandRisk.DESTRUCTIVE, "git push rewrites or deletes remote refs")
        analysis.add(CommandRisk.EXTERNAL, "git push publishes to a remote")
    elif sub in _GIT_NETWORK_SUBCOMMANDS:
        analysis.add(CommandRisk.NETWORK, f"git {sub} contacts a remote")
        if sub == "pull":
            analysis.add(CommandRisk.GIT_WRITE, "git pull updates local branches")
    elif sub == "reset" and "--hard" in flags:
        analysis.add(CommandRisk.DESTRUCTIVE, "git reset --hard discards local changes")
    elif sub == "clean" and any(f.startswith("-") and "f" in f for f in flags):
        analysis.add(CommandRisk.DESTRUCTIVE, "git clean -f deletes untracked files")
    elif sub in {"checkout", "restore"} and "." in flags:
        analysis.add(CommandRisk.DESTRUCTIVE, f"git {sub} . discards working-tree changes")
    elif sub == "branch" and flags & {"-D", "--delete", "-d", "-M", "-m"}:
        analysis.add(
            CommandRisk.DESTRUCTIVE if "-D" in flags else CommandRisk.GIT_WRITE, "git branch deletion/rename"
        )
    elif sub == "stash" and tail[:1] in (["drop"], ["clear"]):
        analysis.add(CommandRisk.DESTRUCTIVE, "git stash drop/clear discards stashed work")
    elif sub == "config" and any("credential" in a for a in tail):
        analysis.add(CommandRisk.SECRET_ACCESS, "git config credential access")
    elif sub in _GIT_READ_SUBCOMMANDS and (
        sub not in {"branch", "tag", "stash", "worktree", "remote", "config"}
        or not tail
        or tail[0]
        in {"list", "-l", "--list", "-a", "-v", "-vv", "show", "--show-current", "--get", "--get-regexp"}
    ):
        analysis.add(CommandRisk.GIT_READ, f"git {sub} (read-only)")
    else:
        analysis.add(CommandRisk.GIT_WRITE, f"git {sub or '?'} modifies the local repository")


def _classify_package_manager(
    prog: str, args: list[str], analysis: CommandAnalysis, workspace: Path | None
) -> None:
    positional = [a for a in args if not a.startswith("-")]
    sub = positional[0] if positional else ""
    sub2 = positional[1] if len(positional) > 1 else ""
    if prog == "make":
        if sub in {"test", "check", "tests", "unittest"}:
            analysis.add(CommandRisk.TEST, f"make {sub}")
        elif sub in {"install", "deploy", "publish", "release", "push"}:
            analysis.add(CommandRisk.UNKNOWN, f"make {sub} may have external effects")
        else:
            analysis.add(CommandRisk.BUILD, f"make {sub or '(default)'}")
        return
    if prog == "npx":
        analysis.add(CommandRisk.NETWORK, "npx may download packages")
        analysis.add(CommandRisk.LOCAL_EXEC, f"npx runs '{sub}'")
        return
    if sub in {"publish", "upload", "deploy", "release"}:
        analysis.add(CommandRisk.EXTERNAL, f"'{prog} {sub}' publishes externally")
        return
    if prog in {"apt", "apt-get", "brew"} and sub in {"install", "remove", "purge", "upgrade", "update"}:
        analysis.add(CommandRisk.PACKAGE_INSTALL, f"'{prog} {sub}' changes system packages")
        analysis.add(CommandRisk.NETWORK, "package download")
        return
    is_test = (
        sub in {"test", "t"}
        or (sub == "run" and sub2.startswith("test"))
        or (prog == "cargo" and sub == "nextest")
        or (prog in {"mvn", "gradle"} and sub in {"test", "verify", "check"})
    )
    if is_test:
        analysis.add(CommandRisk.TEST, f"'{prog} {sub} {sub2}'".replace("  ", " ").replace(" '", "'"))
        return
    if prog in {"uv", "poetry", "pipenv"} and sub == "run" and sub2:
        idx = args.index(sub2) if sub2 in args else len(args)
        _classify_segment(args[idx:], analysis, workspace, 1)
        return
    installs = {
        "install",
        "i",
        "ci",
        "add",
        "sync",
        "update",
        "upgrade",
        "get",
        "fetch",
        "download",
        "remove",
        "uninstall",
        "lock",
    }
    if sub in installs or (prog == "uv" and sub == "pip" and sub2 in {"install", "sync", "uninstall"}):
        analysis.add(CommandRisk.PACKAGE_INSTALL, f"'{prog} {sub}' changes dependencies")
        analysis.add(CommandRisk.NETWORK, "dependency download")
        return
    if sub in {"run", "exec", "x", "dlx"}:
        if sub2 in {"build", "lint", "typecheck", "type-check", "format", "check", "fmt", "tsc"}:
            analysis.add(CommandRisk.BUILD, f"'{prog} {sub} {sub2}'")
        else:
            analysis.add(CommandRisk.LOCAL_EXEC, f"'{prog} {sub} {sub2}' runs a project script")
        return
    if sub in {"build", "check", "vet", "fmt", "clippy", "lint", "tsc"}:
        analysis.add(CommandRisk.BUILD, f"'{prog} {sub}'")
        return
    if (
        sub
        in {
            "list",
            "ls",
            "show",
            "info",
            "tree",
            "outdated",
            "why",
            "env",
            "version",
            "--version",
            "doc",
            "help",
            "config",
        }
        or not sub
    ):
        analysis.add(CommandRisk.READ_ONLY, f"'{prog} {sub}'")
        return
    analysis.add(CommandRisk.UNKNOWN, f"'{prog} {sub}' (unclassified package-manager action)")


def _classify_interpreter(
    prog: str, args: list[str], analysis: CommandAnalysis, workspace: Path | None, depth: int
) -> None:
    if not args:
        analysis.add(CommandRisk.LOCAL_EXEC, f"starts interactive '{prog}'")
        return
    if args[0] in {"--version", "-V"}:
        analysis.add(CommandRisk.READ_ONLY, f"'{prog} --version'")
        return
    if args[0] == "-m" and len(args) > 1:
        module = args[1]
        if module in {"pytest", "unittest", "doctest", "nose2"}:
            analysis.add(CommandRisk.TEST, f"'{prog} -m {module}'")
        elif module == "pip":
            _classify_package_manager("pip", args[2:], analysis, workspace)
        elif module in {
            "mypy",
            "ruff",
            "black",
            "flake8",
            "pylint",
            "compileall",
            "py_compile",
            "isort",
            "pyright",
        }:
            analysis.add(CommandRisk.BUILD, f"'{prog} -m {module}'")
        else:
            analysis.add(CommandRisk.LOCAL_EXEC, f"runs module '{module}'")
        return
    if prog in {"bash", "sh", "zsh"} and args[0] == "-c" and len(args) > 1:
        if depth > 3:
            analysis.add(CommandRisk.UNKNOWN, "deeply nested shell")
            return
        nested = analyze_command(args[1], workspace=workspace, _depth=depth + 1)
        analysis.risks |= nested.risks
        analysis.reasons.extend(r for r in nested.reasons if r not in analysis.reasons)
        analysis.programs.extend(nested.programs)
        return
    if args[0] in {"-c", "-e", "--eval", "-p"}:
        code = args[1] if len(args) > 1 else ""
        if re.search(r"(?i)(urllib|requests|http\.client|socket|fetch\(|https?://)", code):
            analysis.add(CommandRisk.NETWORK, f"inline {prog} code performs network access")
        if re.search(r"(?i)(os\.environ|process\.env|getenv)", code):
            analysis.add(CommandRisk.SECRET_ACCESS, f"inline {prog} code reads environment variables")
        analysis.add(CommandRisk.LOCAL_EXEC, f"runs inline {prog} code")
        return
    script = next((a for a in args if not a.startswith("-")), "")
    if script and workspace is not None and _outside(workspace, script):
        analysis.add(CommandRisk.UNKNOWN, f"runs script outside the workspace '{script}'")
    else:
        analysis.add(CommandRisk.LOCAL_EXEC, f"runs {prog} script '{script}'")


def analyze_command(
    command: str | list[str], *, workspace: Path | None = None, _depth: int = 0
) -> CommandAnalysis:
    analysis = CommandAnalysis()
    if isinstance(command, list):
        segments, ok = [command], True
        text = shlex.join(command)
    else:
        text = command
        segments, ok = _segments(command)
    analysis.parse_ok = ok
    if not ok:
        analysis.add(CommandRisk.UNKNOWN, "command could not be parsed reliably")
    if re.search(r"(curl|wget)[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", text):
        analysis.add(CommandRisk.DESTRUCTIVE, "pipes downloaded content into a shell")
        analysis.add(CommandRisk.NETWORK, "downloads remote content")
    if re.search(r":\(\)\s*\{\s*:\|:&\s*\};:", text):
        analysis.add(CommandRisk.DESTRUCTIVE, "fork bomb")
    if re.search(r"(^|[\s;&|(])eval\s", text):
        analysis.add(CommandRisk.UNKNOWN, "uses eval")
    for seg in segments:
        _classify_segment(seg, analysis, workspace, _depth)
    if not analysis.risks:
        analysis.add(CommandRisk.UNKNOWN, "no recognizable command")
    return analysis
