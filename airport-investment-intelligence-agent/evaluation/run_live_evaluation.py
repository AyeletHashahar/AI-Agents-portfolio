"""Resumable LIVE behavioral evaluation of the real Gemini-powered Agent (10 cases).

    python -m evaluation.run_live_evaluation                       start a new run
    python -m evaluation.run_live_evaluation --resume <run dir>    continue an interrupted run

Each case runs the real AgentSession (production model, budgets, instructions and tools)
with a fresh AgentToolContext; the only Gemini calls are the ones the Agent itself makes.
Nothing is injected into prompts, nothing is pre-resolved and no tool call is forced; the
runner only OBSERVES (execution trace, committed turn, number/size of model requests).

Checkpoints: a case is saved (atomically) the moment it completes, then report.json is
updated. A case interrupted by a crash or power loss restarts from its beginning (setup
included) on --resume; completed cases are skipped and never overwritten.

Provider failures:
- transient (HTTP 429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE, 500/502/504/408, network
  errors): record the failed attempt, persist, wait 30 minutes - or longer if the provider's
  retry hint (RetryInfo.retryDelay / Retry-After) is longer - and retry the SAME case;
- anything else (auth/config, invalid request, programming error, corrupted state): persist,
  print the traceback and stop with exit status 1.
The Agent failing a turn by itself (EmptyModelResponse, FinalizationFailed) is a RESULT of
the case, not an infrastructure error: the case is saved with outcome "agent_error".
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import requests
from google.genai import __version__ as genai_version
from google.genai import errors as genai_errors

import agent.loop as loop
from agent import AgentSession, EmptyModelResponse, FinalizationFailed
from agent_tools import AgentToolContext
from evaluation.cases import CASES, EVALUATION_VERSION
from evaluation.evaluator import evaluate_case, setup_checks
from evaluation.storage import RunStore, RunStoreError

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation_results"
DEFAULT_RETRY_WAIT_S = 30 * 60
HEARTBEAT_S = 5 * 60
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_PROVIDER_STATUSES = frozenset({"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"})
TRANSIENT_NETWORK_ERRORS = (httpx.TransportError, requests.ConnectionError, requests.Timeout, ConnectionError,
                            TimeoutError)
AGENT_RESULT_ERRORS = (EmptyModelResponse, FinalizationFailed)
TRANSIENT, FATAL = "transient", "fatal"
MAX_MESSAGE_CHARS = 2000


# --- error classification and retry delay ------------------------------------------------------

def classify(exc):
    if isinstance(exc, genai_errors.APIError):
        if exc.code in TRANSIENT_HTTP_CODES or exc.status in TRANSIENT_PROVIDER_STATUSES:
            return TRANSIENT
        return FATAL                       # 400 invalid request, 401/403 auth, 404 model, ...
    if isinstance(exc, TRANSIENT_NETWORK_ERRORS):
        return TRANSIENT
    return FATAL


def _seconds(text):
    text = str(text).strip()
    try:
        return float(text[:-1]) if text.endswith("s") else float(text)
    except ValueError:
        return None


def retry_hint_seconds(exc):
    """Longest reliable provider retry hint in seconds, or None: google.rpc.RetryInfo
    'retryDelay' in the error details, or an HTTP Retry-After header (seconds or date)."""
    hints = []
    details = getattr(exc, "details", None)
    error = details.get("error", details) if isinstance(details, dict) else {}
    for item in (error.get("details") or []) if isinstance(error, dict) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo") and "retryDelay" in item:
            seconds = _seconds(item["retryDelay"])
            if seconds is not None:
                hints.append(seconds)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None and hasattr(headers, "get") else None
    if value:
        seconds = _seconds(value)
        if seconds is None:
            try:
                when = parsedate_to_datetime(value)
                seconds = (when - datetime.now(when.tzinfo)).total_seconds()
            except (TypeError, ValueError):
                seconds = None
        if seconds is not None:
            hints.append(seconds)
    return max(hints) if hints else None


def retry_wait_seconds(exc):
    """30 minutes, or the provider's hint when it is longer."""
    hint = retry_hint_seconds(exc)
    return max(DEFAULT_RETRY_WAIT_S, hint) if hint else DEFAULT_RETRY_WAIT_S


# --- observation -------------------------------------------------------------------------------

def serialize_turn(contents):
    """Plain-data transcript of Gemini Content: text, function calls and responses. Thought
    parts and thought signatures are never stored."""
    turn = []
    for content in contents or ():
        parts = []
        for part in content.parts or ():
            if part.thought:
                continue
            if part.function_call:
                parts.append({"function_call": {"id": part.function_call.id, "name": part.function_call.name,
                                                "args": dict(part.function_call.args or {})}})
            elif part.function_response:
                parts.append({"function_response": {"id": part.function_response.id,
                                                    "name": part.function_response.name,
                                                    "response": part.function_response.response}})
            elif part.text:
                parts.append({"text": part.text})
        turn.append({"role": content.role, "parts": parts})
    return turn


class ObservingClient:
    """Pass-through wrapper of the Gemini client: records, per model request, the number of
    Content items and whether the first setup user message is present as text (to show the
    old turn is not in the visible context). Changes nothing; makes no extra request."""

    def __init__(self, client, marker=None):
        self._client, self.marker, self.requests, self.phase = client, marker, [], None
        self.models = self

    def generate_content(self, *, model, contents, config):
        contains = bool(self.marker) and any(self.marker in (part.text or "") for content in contents
                                             for part in (content.parts or ()))
        self.requests.append({"phase": self.phase, "contents": len(contents), "contains_setup_turn_1": contains})
        return self._client.models.generate_content(model=model, contents=contents, config=config)


class Attempt:
    """Everything observed during one attempt of one case (also when it fails midway).
    A fresh AgentSession and AgentToolContext per attempt: cases never share state."""

    def __init__(self, case, client, context_factory=AgentToolContext):
        self.case = case
        self.observer = ObservingClient(client, case.setup[0] if case.setup else None)
        self.session = AgentSession(client=self.observer, context=context_factory())
        self.setup, self.phase, self.gemini_calls = [], None, 0

    def send(self, message, phase):
        """One real Agent turn. Gemini calls are counted from its execution trace, whether
        the turn succeeds, fails as an Agent result, or raises."""
        self.phase = self.observer.phase = phase
        try:
            result = "completed", self.session.send(message), None
        except AGENT_RESULT_ERRORS as exc:
            result = "agent_error", None, {"type": type(exc).__name__, "message": str(exc)[:MAX_MESSAGE_CHARS]}
        finally:
            self.gemini_calls += self.session.last_trace.total_llm_calls if self.session.last_trace else 0
        return result

    def turn(self):
        return serialize_turn(self.session._turns[-1]) if self.session._turns else None


def execute_case(case, attempt):
    """Run one case (setup turns, then the evaluated turn). Infrastructure/provider errors
    propagate; Agent-level failures become the case outcome."""
    for i, message in enumerate(case.setup, 1):
        before = len(attempt.session._turns)
        outcome, answer, error = attempt.send(message, f"setup_{i}")
        attempt.setup.append({"prompt": message, "outcome": outcome, "final_answer": answer, "error": error,
                              "trace": attempt.session.last_trace.to_dict(),
                              "turn": attempt.turn() if len(attempt.session._turns) > before else None})
        if outcome != "completed":
            break
    setup_valid, setup_check_items = (True, []) if not case.setup else setup_checks(case, attempt.setup)
    record = {"setup": attempt.setup, "setup_checks": setup_check_items, "final_answer": None, "trace": None,
              "turn": None, "error": None, "evaluated_requests": []}
    if not setup_valid:
        record["outcome"] = "setup_invalid"
        return record
    turns_before = len(attempt.session._turns)
    outcome, answer, error = attempt.send(case.prompt, "evaluated")
    record.update(outcome=outcome, final_answer=answer, error=error, trace=attempt.session.last_trace.to_dict(),
                  turn=attempt.turn() if len(attempt.session._turns) > turns_before else None,
                  evaluated_requests=[r for r in attempt.observer.requests if r["phase"] == "evaluated"],
                  visible_turns_at_evaluation=min(attempt.session.recent_turns, turns_before),
                  stored_turns_before_evaluation=turns_before)
    return record


# --- runner ------------------------------------------------------------------------------------

class Runner:
    def __init__(self, store, client_factory=loop.create_gemini_client, sleep=time.sleep, now=datetime.now,
                 out=print, clock=time.perf_counter, context_factory=AgentToolContext):
        self.store, self.client_factory, self.sleep, self.now, self.out, self.clock = \
            store, client_factory, sleep, now, out, clock
        self.context_factory = context_factory
        self.client = None

    def log(self, message):
        self.out(f"[{self.now():%H:%M}] {message}", flush=True)

    def run(self):
        done = self.store.completed_ids()
        total = len(CASES)
        try:
            self.client = self.client_factory()
        except Exception as exc:
            return self._fatal(None, exc, {"phase": "client_creation"})
        for index, case in enumerate(CASES, 1):
            if case.id in done:
                self.log(f"Case {case.id}/{total:02d} — {case.name}: already completed, skipped")
                continue
            status = self._run_case(index, total, case)
            if status is not None:
                return status
        self.store.mark_completed()
        report = self.store.build_report()
        self.log(f"FINISHED — {len(report['completed_cases'])}/{total} cases, "
                 f"{report['total_gemini_calls_including_failed_attempts']} Gemini calls in total")
        self.log(f"Report: {self.store.directory / 'report.json'}")
        return 0

    def _run_case(self, index, total, case):
        while True:
            number = self.store.begin_attempt(case.id)
            self.log(f"Case {case.id}/{total:02d} — {case.name}" + (f" (attempt {number})" if number > 1 else ""))
            attempt = Attempt(case, self.client, self.context_factory)
            started_at, start = self.now(), self.clock()
            try:
                record = execute_case(case, attempt)
                result = self._result(case, record, attempt, started_at, self.clock() - start, number)
            except Exception as exc:
                info = self._failure(exc, attempt, started_at, number)
                if info["classification"] == FATAL:
                    return self._fatal(case, exc, info)
                self.store.record_failed_attempt(case.id, info)
                self.log(f"Gemini {info['status_code'] or ''} {info['provider_status'] or info['error_type']} "
                         f"— transient (attempt {number}, {info['gemini_calls']} Gemini calls used)")
                self.log("Progress saved")
                self._wait(info["wait_s"])
                self.log(f"Retrying case {case.id}")
                continue
            self.log(f"Completed — {result['outcome']}, {result['gemini_calls']['total']} Gemini calls")
            try:
                path = self.store.save_case(case.id, result)
            except Exception as exc:
                return self._fatal(case, exc, {"phase": "save"})
            self.log(f"Saved {path}")
            return None

    def _result(self, case, record, attempt, started_at, duration, number):
        evaluation = evaluate_case(case, record)
        pending = self.store.pending_attempts(case.id)
        failed_calls = sum(a.get("gemini_calls") or 0 for a in pending["failed_attempts"])
        setup_calls = sum((s["trace"] or {}).get("total_llm_calls", 0) for s in record["setup"])
        evaluated_calls = (record["trace"] or {}).get("total_llm_calls", 0)
        return {
            "evaluation_version": EVALUATION_VERSION, "run_id": self.store.run["run_id"],
            "case_id": case.id, "case_name": case.name, "kind": case.kind, "prompt": case.prompt,
            "setup_description": case.setup_description or None, "setup_prompts": list(case.setup),
            "expectations": list(case.expectations), "review_focus": list(case.review_focus),
            "outcome": record["outcome"], "final_answer": record["final_answer"], "error": record["error"],
            "trace": record["trace"], "turn_transcript": record["turn"],
            "setup": record["setup"], "setup_checks": record["setup_checks"],
            "evaluated_requests": record["evaluated_requests"],
            "visible_turns_at_evaluation": record.get("visible_turns_at_evaluation"),
            "stored_turns_before_evaluation": record.get("stored_turns_before_evaluation"),
            **evaluation,
            "gemini_calls": {"setup": setup_calls, "evaluated_turn": evaluated_calls,
                             "total": setup_calls + evaluated_calls, "failed_attempts": failed_calls,
                             "observed_requests": len(attempt.observer.requests)},
            "tool_rounds": (record["trace"] or {}).get("tool_rounds"),
            "retry_count": number - 1,
            "failed_attempts": pending["failed_attempts"],
            "interrupted_attempts": max(number - 1 - len(pending["failed_attempts"]), 0),
            "started_at": started_at.isoformat(timespec="seconds"),
            "completed_at": self.now().isoformat(timespec="seconds"),
            "duration_s": round(duration, 3),
            "model": attempt.session.model,          # the model this case's AgentSession actually used
        }

    def _failure(self, exc, attempt, started_at, number):
        classification = classify(exc)
        hint = retry_hint_seconds(exc) if classification == TRANSIENT else None
        return {"attempt": number, "phase": attempt.phase, "started_at": started_at.isoformat(timespec="seconds"),
                "failed_at": self.now().isoformat(timespec="seconds"), "classification": classification,
                "error_type": type(exc).__name__, "message": str(exc)[:MAX_MESSAGE_CHARS],
                "status_code": getattr(exc, "code", None) if isinstance(exc, genai_errors.APIError) else None,
                "provider_status": getattr(exc, "status", None) if isinstance(exc, genai_errors.APIError) else None,
                "retry_hint_s": hint, "wait_s": retry_wait_seconds(exc) if classification == TRANSIENT else None,
                "gemini_calls": attempt.gemini_calls,
                "trace": attempt.session.last_trace.to_dict() if attempt.session.last_trace else None,
                "traceback": traceback.format_exc() if classification == FATAL else None}

    def _wait(self, seconds):
        resume_at = self.now() + timedelta(seconds=seconds)
        self.log(f"Waiting — retrying at approximately {resume_at:%H:%M} ({seconds / 60:.0f} min)")
        remaining = seconds
        while remaining > 0:
            chunk = min(HEARTBEAT_S, remaining)
            self.sleep(chunk)
            remaining -= chunk
            if remaining > 0:
                self.log(f"Still waiting (intentionally) — retry at approximately {resume_at:%H:%M}")

    def _fatal(self, case, exc, info):
        info = {**info, "case_id": case.id if case else None, "error_type": type(exc).__name__,
                "message": str(exc)[:MAX_MESSAGE_CHARS], "traceback": traceback.format_exc()}
        if case is not None:
            self.store.record_failed_attempt(case.id, info)
        self.store.mark_fatal(info)
        self.log(f"STOPPED — fatal {type(exc).__name__}" + (f" in case {case.id}" if case else "") + f": {exc}")
        self.log(f"Progress saved in {self.store.directory}")
        print(info["traceback"], file=sys.stderr, flush=True)
        return 1


def main(argv=None, *, client_factory=loop.create_gemini_client, sleep=time.sleep, now=datetime.now, out=print,
         results_dir=RESULTS_DIR, context_factory=AgentToolContext):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run directory")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            store = RunStore.open(args.resume, now=now)
            store.mark_running(resumed=True)
        else:
            store = RunStore.create(results_dir, {"model": loop.MODEL, "recent_turns": loop.RECENT_TURNS,
                                                  "max_tool_rounds": loop.MAX_TOOL_ROUNDS,
                                                  "max_llm_calls": loop.MAX_LLM_CALLS,
                                                  "google_genai_version": genai_version}, now=now)
    except RunStoreError as exc:
        out(f"[{now():%H:%M}] Cannot use the run directory: {exc}", flush=True)
        return 2
    runner = Runner(store, client_factory=client_factory, sleep=sleep, now=now, out=out,
                    context_factory=context_factory)
    runner.log(("Resuming" if args.resume else "Starting") + f" live evaluation {store.run['run_id']} "
               f"({EVALUATION_VERSION}) — results in {store.directory}")
    try:
        return runner.run()
    except KeyboardInterrupt:
        runner.log(f"Interrupted. Completed cases are saved; resume with: "
                   f"python -m evaluation.run_live_evaluation --resume \"{store.directory}\"")
        return 130


if __name__ == "__main__":
    sys.exit(main())
