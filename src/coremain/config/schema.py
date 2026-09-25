"""Configuration schema. Every key is typed and validated; unknown keys are errors."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coremain.providers.errors import ProviderErrorClass

Decision = Literal["allow", "ask", "deny"]
ROLE_NAMES = ("planner", "implementer", "researcher", "reviewer", "debugger", "verifier", "summarizer")
STRENGTH_DIMENSIONS = (
    "coding", "planning", "review", "research", "debugging", "ui", "security", "writing", "long_context", "tool_use",
)
CRED_REF_RE = re.compile(r"^(env|file|command|store):.+$")


def _cred_ref(value: str | None) -> str | None:
    if value is None:
        return None
    if not CRED_REF_RE.match(value):
        raise ValueError(
            "must be a credential reference such as 'env:OPENAI_API_KEY', 'store:openai', "
            "'file:/path/to/key' or 'command:pass show openai'; literal secrets are not allowed in config files"
        )
    return value


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class ProviderConfig(_Base):
    kind: Literal["openai_compatible", "anthropic", "scripted"]
    base_url: str | None = None
    api_key: str | None = None
    secret_headers: dict[str, str] = {}
    headers: dict[str, str] = {}
    connect_timeout_s: float = Field(10.0, gt=0)
    read_timeout_s: float = Field(300.0, gt=0)
    max_retries: int = Field(2, ge=0, le=6)
    enabled: bool = True
    script: str | None = None
    extra_body: dict[str, Any] = {}
    description: str | None = None

    _v_key = field_validator("api_key")(classmethod(lambda cls, v: _cred_ref(v)))

    @field_validator("secret_headers")
    @classmethod
    def _v_secret_headers(cls, v: dict[str, str]) -> dict[str, str]:
        for value in v.values():
            _cred_ref(value)
        return v

    @model_validator(mode="after")
    def _v_kind(self) -> ProviderConfig:
        if self.kind == "scripted" and not self.script:
            raise ValueError("scripted providers require 'script' (path to a JSON script file)")
        if self.kind == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible providers require 'base_url'")
        return self


class ModelCapabilities(_Base):
    tool_calling: bool = True
    parallel_tool_calls: bool = True
    structured_output: Literal["none", "json_object", "json_schema"] = "none"
    vision: bool = False
    reasoning: Literal["none", "standard", "extended"] = "standard"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    prompt_caching: bool = False
    streaming: bool = True
    embeddings: bool = False


class ModelCost(_Base):
    input_per_mtok: float | None = Field(None, ge=0)
    output_per_mtok: float | None = Field(None, ge=0)
    cached_input_per_mtok: float | None = Field(None, ge=0)
    currency: str = "USD"


class ModelConfig(_Base):
    provider: str
    id: str = Field(min_length=1, description="Exact provider model identifier. Never guessed.")
    display_name: str | None = None
    context_window: int = Field(ge=1024)
    max_output_tokens: int = Field(ge=256)
    capabilities: ModelCapabilities = ModelCapabilities()
    strengths: dict[str, float] = {}
    tier: Literal["frontier", "strong", "standard", "light"] = "standard"
    latency: Literal["fast", "standard", "slow"] = "standard"
    cost: ModelCost = ModelCost()
    roles: list[str] = []
    enabled: bool = True
    params: dict[str, Any] = {}

    @field_validator("strengths")
    @classmethod
    def _v_strengths(cls, v: dict[str, float]) -> dict[str, float]:
        for key, score in v.items():
            if key not in STRENGTH_DIMENSIONS:
                raise ValueError(f"unknown strength '{key}'; expected one of {', '.join(STRENGTH_DIMENSIONS)}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"strength '{key}' must be between 0 and 1")
        return v

    @field_validator("roles")
    @classmethod
    def _v_roles(cls, v: list[str]) -> list[str]:
        for role in v:
            if role not in ROLE_NAMES:
                raise ValueError(f"unknown role '{role}'; expected one of {', '.join(ROLE_NAMES)}")
        return v


class FallbackRule(_Base):
    on: list[ProviderErrorClass]
    to: list[str] = Field(min_length=1)


class RoutingConfig(_Base):
    prefer: Literal["quality", "balanced", "speed", "cost"] = "balanced"
    default_model: str | None = None
    pins: dict[str, str] = {}
    allowed: list[str] | None = None
    fallback: list[FallbackRule] = []
    independent_review: Literal["prefer", "require", "off"] = "prefer"
    max_parallel_proposals: int = Field(2, ge=1, le=4)
    use_history: bool = True

    @field_validator("pins")
    @classmethod
    def _v_pins(cls, v: dict[str, str]) -> dict[str, str]:
        for role in v:
            if role not in ROLE_NAMES:
                raise ValueError(f"unknown role '{role}' in routing.pins")
        return v


class BudgetConfig(_Base):
    max_turns_per_node: int = Field(40, ge=1)
    max_turns_per_task: int = Field(160, ge=1)
    max_tokens_per_task: int = Field(4_000_000, ge=1000)
    max_wall_time_s: int = Field(5400, ge=10)
    max_workers: int = Field(3, ge=1, le=16)
    max_subtask_depth: int = Field(2, ge=0, le=5)
    max_fanout: int = Field(4, ge=1, le=12)
    max_repair_iterations: int = Field(2, ge=0, le=6)
    max_model_retries: int = Field(3, ge=0, le=8)
    max_structured_repairs: int = Field(2, ge=0, le=4)
    max_attempts: int = Field(3, ge=1, le=10)
    max_cost_usd_per_task: float | None = Field(None, ge=0)


class PermissionRule(_Base):
    capability: str
    pattern: str = "*"
    decision: Decision
    reason: str | None = None


class NetworkConfig(_Base):
    offline: bool = False
    allow_domains: list[str] = [
        "context7.com", "*.context7.com", "docs.python.org", "pypi.org", "developer.mozilla.org", "nodejs.org",
        "registry.npmjs.org", "www.npmjs.com", "github.com", "raw.githubusercontent.com", "api.github.com",
        "docs.rs", "pkg.go.dev", "*.readthedocs.io", "localhost", "127.0.0.1",
    ]
    deny_domains: list[str] = []


class PermissionsConfig(_Base):
    profile: Literal["read-only", "standard", "autonomous"] = "standard"
    rules: list[PermissionRule] = []
    network: NetworkConfig = NetworkConfig()
    env_passthrough: list[str] = []
    sensitive_paths: list[str] = []
    non_interactive_ask: Literal["deny", "suspend"] = "suspend"


class WorkspaceConfig(_Base):
    mode: Literal["isolated", "direct"] = "isolated"
    auto_apply: bool = True
    share_paths: list[str] = ["node_modules", ".venv", "venv"]
    keep_days: int = Field(14, ge=0)
    copy_max_mb: int = Field(300, ge=1)


class VerificationConfig(_Base):
    min_level: Literal["weak", "moderate", "strong"] = "weak"
    commands: dict[str, str] = {}
    timeout_s: int = Field(900, ge=5)
    require_review: bool = True
    run_lint: bool = True
    run_typecheck: bool = True

    @field_validator("commands")
    @classmethod
    def _v_commands(cls, v: dict[str, str]) -> dict[str, str]:
        allowed = {"test", "lint", "typecheck", "build", "format"}
        for key in v:
            if key not in allowed:
                raise ValueError(f"verification.commands key '{key}' must be one of {sorted(allowed)}")
        return v


class ContextConfig(_Base):
    budget_fraction: float = Field(0.45, gt=0.05, le=0.9)
    max_budget_tokens: int = Field(120_000, ge=2000)
    max_file_tokens: int = Field(6000, ge=200)
    instruction_files: list[str] = ["AGENTS.md", "CORE.md", "CLAUDE.md", ".core/instructions.md"]
    snapshot_content: bool = True
    compaction_threshold: float = Field(0.72, gt=0.3, lt=0.95)


class MemoryConfig(_Base):
    enabled: bool = True
    global_enabled: bool = False
    max_items_in_context: int = Field(12, ge=0, le=100)


class SkillsConfig(_Base):
    enabled: bool = True
    max_selected: int = Field(3, ge=0, le=10)
    max_tokens: int = Field(6000, ge=200)
    allow_untrusted: bool = False
    disabled: list[str] = []


class MCPServerConfig(_Base):
    transport: Literal["stdio", "http"]
    command: list[str] = []
    env: dict[str, str] = {}
    secret_env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, str] = {}
    secret_headers: dict[str, str] = {}
    enabled: bool = True
    trust: Literal["untrusted", "trusted"] = "untrusted"
    timeout_s: float = Field(60.0, gt=0)
    startup_timeout_s: float = Field(30.0, gt=0)
    tool_allow: list[str] = ["*"]
    tool_deny: list[str] = []
    description: str | None = None

    @model_validator(mode="after")
    def _v(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError("http MCP servers require 'url'")
        for value in (*self.secret_env.values(), *self.secret_headers.values()):
            _cred_ref(value)
        return self


class BrowserConfig(_Base):
    enabled: bool = True
    executable_path: str | None = None
    headless: bool = True
    allowed_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*", "file://*"]
    launch_args: list[str] = []
    navigation_timeout_s: float = Field(30.0, gt=0)


class ResearchConfig(_Base):
    docs_provider: Literal["context7", "none"] = "context7"
    context7_api_key: str | None = None
    context7_base_url: str = "https://context7.com/api"
    max_fetch_bytes: int = Field(2_000_000, ge=1000)
    cache_ttl_s: int = Field(86_400, ge=0)

    _v_key = field_validator("context7_api_key")(classmethod(lambda cls, v: _cred_ref(v)))


class ExecConfig(_Base):
    default_timeout_s: float = Field(300.0, gt=0)
    max_timeout_s: float = Field(3600.0, gt=0)
    max_output_bytes: int = Field(4_000_000, ge=10_000)
    kill_grace_s: float = Field(3.0, ge=0)
    max_memory_mb: int | None = Field(None, ge=64)


class UIConfig(_Base):
    theme: Literal["dark", "light"] = "dark"
    show_decisions: bool = True
    stream_fps: int = Field(20, ge=1, le=60)


class LearningConfig(_Base):
    enabled: bool = True
    record_traces: bool = True
    auto_propose_heuristics: bool = True


class LSPServerConfig(_Base):
    command: list[str]
    languages: list[str]
    root_markers: list[str] = []


class IntelConfig(_Base):
    max_file_bytes: int = Field(512_000, ge=1000)
    max_files: int = Field(60_000, ge=10)
    exclude: list[str] = []
    lsp_enabled: bool = True
    lsp_servers: dict[str, LSPServerConfig] = {}


class CoreConfig(_Base):
    providers: dict[str, ProviderConfig] = {}
    models: dict[str, ModelConfig] = {}
    routing: RoutingConfig = RoutingConfig()
    budgets: BudgetConfig = BudgetConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    verification: VerificationConfig = VerificationConfig()
    context: ContextConfig = ContextConfig()
    memory: MemoryConfig = MemoryConfig()
    skills: SkillsConfig = SkillsConfig()
    mcp: dict[str, MCPServerConfig] = {}
    browser: BrowserConfig = BrowserConfig()
    research: ResearchConfig = ResearchConfig()
    exec: ExecConfig = ExecConfig()
    ui: UIConfig = UIConfig()
    learning: LearningConfig = LearningConfig()
    intel: IntelConfig = IntelConfig()

    @model_validator(mode="after")
    def _cross_references(self) -> CoreConfig:
        for key, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model '{key}' references unknown provider '{model.provider}'")
        known = set(self.models)
        refs: list[tuple[str, str]] = []
        if self.routing.default_model:
            refs.append(("routing.default_model", self.routing.default_model))
        refs.extend((f"routing.pins.{r}", m) for r, m in self.routing.pins.items())
        for i, rule in enumerate(self.routing.fallback):
            refs.extend((f"routing.fallback[{i}].to", m) for m in rule.to)
        refs.extend(("routing.allowed", m) for m in (self.routing.allowed or []))
        for where, ref in refs:
            if ref not in known:
                raise ValueError(f"{where} references unknown model '{ref}' (configured: {sorted(known) or 'none'})")
        return self
