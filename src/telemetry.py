from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunTelemetryCounter:
    """Per-run counters plus an ordered trace that the UI renders as the agent timeline."""

    llm_calls: int = 0
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    external_api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_ms: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def record_llm_call(self, agent: str = "agent", *, input_tokens: int = 0, output_tokens: int = 0,
                        ms: float = 0.0, stop_reason: str | None = None) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_ms += ms
        self.event("llm", agent=agent, input_tokens=input_tokens, output_tokens=output_tokens,
                   ms=round(ms), stop_reason=stop_reason)

    def record_tool_call(self, name: str, *, caller: str = "agent", args: dict | None = None, ok: bool = True) -> None:
        self.tool_calls += 1
        self.tool_names.append(name)
        self.event("tool", name=name, caller=caller, args=args or {}, ok=ok)

    def record_external_call(self, name: str, ok: bool, detail: str = "") -> None:
        self.external_api_calls += 1
        self.event("external", name=name, ok=ok, detail=detail)

    def event(self, kind: str, **data: Any) -> None:
        self.trace.append({"t_ms": round((time.perf_counter() - self._t0) * 1000), "kind": kind, **data})
