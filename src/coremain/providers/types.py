"""Provider-agnostic request/response and streaming types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str = ""
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(d["id"], d["name"], d.get("arguments"), d.get("raw_arguments", ""), d.get("parse_error"))


@dataclass
class ImageBlock:
    media_type: str
    data_b64: str


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    images: list[ImageBlock] = field(default_factory=list)
    # Opaque provider continuation state (e.g. reasoning signatures). Kept in memory only:
    # never persisted, displayed or sent to a different provider.
    provider_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for checkpoints (provider_state is deliberately dropped)."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "is_error": self.is_error,
            "images": [{"media_type": i.media_type, "data_b64": i.data_b64} for i in self.images],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatMessage:
        return cls(
            d["role"],
            d.get("content", ""),
            [ToolCall.from_dict(t) for t in d.get("tool_calls", [])],
            d.get("tool_call_id"),
            d.get("name"),
            d.get("is_error", False),
            [ImageBlock(i["media_type"], i["data_b64"]) for i in d.get("images", [])],
        )


@dataclass
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] = field(default_factory=list)
    tool_choice: Literal["auto", "required", "none"] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    response_schema: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cache_write_tokens: int = 0
    provider_cost_usd: float | None = None

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_write_tokens += other.cache_write_tokens
        if other.provider_cost_usd is not None:
            self.provider_cost_usd = (self.provider_cost_usd or 0.0) + other.provider_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    chars: int


@dataclass
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass
class ToolCallDelta:
    index: int
    arguments: str


@dataclass
class UsageUpdate:
    usage: Usage


@dataclass
class Finish:
    reason: str  # stop | tool_calls | length | content_filter | other
    raw: str | None = None


@dataclass
class ProviderStateUpdate:
    state: dict[str, Any]


StreamEvent = (
    TextDelta | ReasoningDelta | ToolCallStart | ToolCallDelta | UsageUpdate | Finish | ProviderStateUpdate
)


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str
    provider_state: dict[str, Any] | None = None
    latency_ms: int = 0
    ttft_ms: int | None = None
    model_ref: str = ""
    model_call_id: str | None = None
    retries: int = 0


@dataclass
class DiscoveredModel:
    id: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("raw", None)
        return d
