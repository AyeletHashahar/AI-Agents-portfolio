"""Minimal structured execution trace of ONE agent turn (instrumentation only).

Records HOW a turn executed - model calls, tool calls, budgets, timing, failure - never
WHAT the model thought or said: no thought parts, no model text, no full tool or memory
results, no conversation history. Tool arguments are kept (needed for evaluation), with
very long strings shortened.

    TurnRecorder   mutable, filled in by agent/loop.py while the turn runs
    TurnTrace      frozen result of TurnRecorder.finish(); exposed as AgentSession.last_trace

Timing uses time.perf_counter() (monotonic). Everything stays in memory.
"""

import copy
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from agent.memory import MEMORY_TOOL_NAME

MAX_TRACE_STRING_CHARS = 500    # longer argument strings / error messages are shortened

# LLM call outcomes
FINAL_TEXT = "final_text"              # normal call answered with text
FUNCTION_CALLS = "function_calls"      # normal call requested tool(s)
FINALIZATION = "finalization"          # terminal finalization call answered with text
ERROR = "error"                        # request raised, or the response was unusable


@dataclass(frozen=True)
class LLMCallTrace:
    number: int                          # 1-based, in request order
    phase: str                           # "normal" | "finalization"
    outcome: str                         # FINAL_TEXT | FUNCTION_CALLS | FINALIZATION | ERROR
    latency_s: float
    requested_functions: tuple = ()      # function names the response asked for, in order


@dataclass(frozen=True)
class ToolCallTrace:
    round: int                           # 1-based tool round of this turn
    call_id: str | None
    name: str
    category: str                        # "memory" | "business"
    arguments: dict                      # deep copy (long strings shortened)
    latency_s: float
    status: str | None                   # the result's "status" ("ok", "partial", "error"...);
                                         # "exception" if the call raised
    error_codes: tuple = ()              # codes of the result's structured errors
    memory_match_count: int | None = None     # memory calls only
    memory_matched_turns: tuple | None = None  # memory calls only: matched turn numbers


@dataclass(frozen=True)
class RequestedCall:
    """A tool call the model requested but that was NOT executed (budget exhausted)."""
    name: str
    arguments: dict


@dataclass(frozen=True)
class TurnTrace:
    status: str                          # "success" | "error"
    latency_s: float
    llm_calls: tuple                     # LLMCallTrace, in order (finalization included)
    tool_calls: tuple                    # ToolCallTrace, in execution order
    tool_rounds: int                     # completed tool rounds (the loop's budget counter)
    unexecuted_calls: tuple = ()         # RequestedCall, when a budget forced finalization
    error_type: str | None = None
    error_message: str | None = None

    @property
    def total_llm_calls(self):
        return len(self.llm_calls)

    @property
    def normal_llm_calls(self):
        return sum(call.phase == "normal" for call in self.llm_calls)

    @property
    def finalization_used(self):
        return any(call.phase == "finalization" for call in self.llm_calls)

    def to_dict(self):
        """Plain, JSON-serialisable copy (derived counts included)."""
        return {**asdict(self), "total_llm_calls": self.total_llm_calls,
                "normal_llm_calls": self.normal_llm_calls, "finalization_used": self.finalization_used}


def _compact(value):
    """Deep copy of tool arguments with long strings shortened, so the trace stays small."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TRACE_STRING_CHARS else value[:MAX_TRACE_STRING_CHARS] + " ...[truncated]"
    if isinstance(value, Mapping):
        return {str(k): _compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(v) for v in value]
    return copy.deepcopy(value)


class TurnRecorder:
    """Collects one turn's trace. The loop reports events; nothing here affects execution."""

    def __init__(self):
        self._start = time.perf_counter()
        self._llm_calls = []
        self._tool_calls = []
        self._unexecuted = []
        self._tool_rounds = 0
        self._pending = None              # (phase, start) of the model request in flight

    # --- model calls -------------------------------------------------------------------
    def llm_started(self, phase):
        self._pending = (phase, time.perf_counter())

    def llm_finished(self, outcome, functions=()):
        phase, start = self._pending
        self._pending = None
        self._llm_calls.append(LLMCallTrace(len(self._llm_calls) + 1, phase, outcome,
                                            time.perf_counter() - start, tuple(functions)))

    # --- tool calls --------------------------------------------------------------------
    def tool_called(self, call, latency_s, result=None, exception=None):
        memory = call.name == MEMORY_TOOL_NAME
        result = result if isinstance(result, Mapping) else {}
        matches = result.get("matches") if memory else None
        self._tool_calls.append(ToolCallTrace(
            round=self._tool_rounds + 1, call_id=call.id, name=call.name,
            category="memory" if memory else "business", arguments=_compact(dict(call.args or {})),
            latency_s=latency_s, status="exception" if exception is not None else result.get("status"),
            error_codes=tuple(e.get("code") for e in result.get("errors") or [] if isinstance(e, Mapping)),
            memory_match_count=len(matches) if isinstance(matches, list) else None,
            memory_matched_turns=tuple(m.get("turn") for m in matches) if isinstance(matches, list) else None))

    def round_completed(self):
        self._tool_rounds += 1

    def not_executed(self, calls):
        self._unexecuted.extend(RequestedCall(c.name, _compact(dict(c.args or {}))) for c in calls)

    # --- end of turn -------------------------------------------------------------------
    def finish(self, error=None):
        """Freeze the trace. A model request still in flight when `error` happened is
        recorded as an ERROR call."""
        if self._pending is not None:
            self.llm_finished(ERROR)
        message = None if error is None else _compact(str(error))
        return TurnTrace(status="error" if error is not None else "success",
                         latency_s=time.perf_counter() - self._start,
                         llm_calls=tuple(self._llm_calls), tool_calls=tuple(self._tool_calls),
                         tool_rounds=self._tool_rounds, unexecuted_calls=tuple(self._unexecuted),
                         error_type=None if error is None else type(error).__name__,
                         error_message=message)
