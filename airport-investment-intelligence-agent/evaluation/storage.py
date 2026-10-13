"""Persistent, resumable storage of one live-evaluation run (local JSON files only).

    evaluation_results/run_<YYYYmmdd-HHMMSS>/
        run.json          run metadata + state of the case in progress (attempts, failures)
        case_<id>.json    one COMPLETED case; written once, never overwritten
        report.json       aggregate report, rebuilt after every state change

Resume is allowed only if the evaluated Agent contract - the Gemini model and everything Gemini
sees - is unchanged (see contract_components): otherwise results of two Agent versions would be
mixed in one run.

Every write is atomic (temporary file in the same directory -> flush + fsync -> os.replace),
so a crash leaves either the previous or the new version of a file, never a partial one.
A case file exists if and only if that case completed. Checkpoint granularity is the
completed case: nothing of a half-finished case is persisted except attempt metadata.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import agent.loop as loop
from evaluation.cases import CASE_IDS, CASES, EVALUATION_VERSION
from evaluation.evaluator import DIMENSIONS

RUN_FILE, REPORT_FILE = "run.json", "report.json"
RUNNING, COMPLETED, STOPPED_FATAL = "running", "completed", "stopped_fatal"
_REPLACE_ATTEMPTS = 10          # os.replace can briefly fail on Windows while a synced/scanned file is open


class RunStoreError(Exception):
    """The run directory cannot be used (missing, corrupted or incompatible). Nothing is changed."""


class CaseAlreadyCompleted(RunStoreError):
    """Refusing to overwrite the result of a completed case."""


def _hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_components():
    """Hashes of what the evaluated Agent is, as Gemini sees it, plus what is evaluated:
      model                    - the Gemini model the Agent calls (agent.loop.MODEL);
      cases                    - the case definitions (prompts, setup, expectations);
      system_instruction       - the system instruction of the normal request config;
      tool_declarations        - every Gemini-visible function declaration (name, description,
                                 parameter schema), in the order sent;
      finalization_instruction - the text sent when a budget forces the final answer.
    Taken from the request config the Agent actually builds (agent.loop.build_config), not
    from source files, so refactors that do not change what Gemini sees keep runs resumable."""
    config = loop.build_config()
    declarations = [declaration.model_dump(mode="json", exclude_none=True)
                    for tool in config.tools for declaration in tool.function_declarations]
    return {"model": _hash(loop.MODEL),
            "cases": _hash([asdict(case) for case in CASES]),
            "system_instruction": _hash(config.system_instruction),
            "tool_declarations": _hash(declarations),
            "finalization_instruction": _hash(loop.FINALIZATION_INSTRUCTION)}


def contract_fingerprint(components=None):
    return _hash(components if components is not None else contract_components())


def atomic_write_json(path, data, sleep=time.sleep):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            sleep(0.2)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise RunStoreError(f"Cannot read {path}: {type(exc).__name__}: {exc}") from exc


def _timestamp(now):
    return now().isoformat(timespec="seconds")


class RunStore:
    def __init__(self, directory, run, now=datetime.now):
        self.directory, self.run, self.now = Path(directory), run, now

    # --- creation / resume ------------------------------------------------------------------
    @classmethod
    def create(cls, results_dir, metadata, now=datetime.now):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = now().strftime("%Y%m%d-%H%M%S")
        directory, n = results_dir / f"run_{stamp}", 1
        while directory.exists():
            n += 1
            directory = results_dir / f"run_{stamp}_{n}"
        directory.mkdir()
        components = contract_components()
        run = {"evaluation_version": EVALUATION_VERSION, "contract_fingerprint": contract_fingerprint(components),
               "contract_components": components,
               "run_id": directory.name, "case_ids": list(CASE_IDS), "status": RUNNING,
               "started_at": _timestamp(now), "updated_at": _timestamp(now), "resumed_at": [],
               "metadata": metadata, "pending_case": None, "fatal_error": None}
        store = cls(directory, run, now)
        store._persist()
        return store

    @classmethod
    def open(cls, directory, now=datetime.now):
        """Load and validate an existing run for resuming. Raises RunStoreError (and changes
        nothing) if it is missing, corrupted or from a different evaluation definition."""
        directory = Path(directory)
        if not (directory / RUN_FILE).is_file():
            raise RunStoreError(f"No {RUN_FILE} in {directory}.")
        run = _read_json(directory / RUN_FILE)
        required = {"evaluation_version", "contract_fingerprint", "contract_components", "run_id", "case_ids",
                    "status", "started_at", "pending_case", "metadata"}
        if isinstance(run, dict) and "evaluation_version" in run and run["evaluation_version"] != EVALUATION_VERSION:
            raise RunStoreError(f"Evaluation version mismatch: run is {run['evaluation_version']!r}, "
                                f"this code is {EVALUATION_VERSION!r}. Start a new run.")
        if not isinstance(run, dict) or not required <= set(run):
            raise RunStoreError(f"{RUN_FILE} is corrupted or incomplete (missing {sorted(required - set(run or {}))}).")
        current = contract_components()
        if run["contract_fingerprint"] != contract_fingerprint(current) or run["case_ids"] != list(CASE_IDS):
            stored = run["contract_components"] if isinstance(run["contract_components"], dict) else {}
            changed = sorted(name for name, value in current.items() if stored.get(name) != value)
            raise RunStoreError(f"The evaluated Agent contract changed since this run started "
                                f"({', '.join(changed) or 'fingerprint'}); resuming would mix results of different "
                                f"Agent versions. Start a new run.")
        pending = run["pending_case"]
        if pending is not None and (not isinstance(pending, dict) or pending.get("case_id") not in CASE_IDS):
            raise RunStoreError(f"{RUN_FILE} has an invalid pending_case.")
        store = cls(directory, run, now)
        for case_id in store.completed_ids():
            result = _read_json(store.case_path(case_id))
            if (not isinstance(result, dict) or result.get("case_id") != case_id
                    or result.get("evaluation_version") != EVALUATION_VERSION or result.get("run_id") != run["run_id"]):
                raise RunStoreError(f"{store.case_path(case_id).name} does not belong to this run.")
        return store

    # --- queries ------------------------------------------------------------------------------
    def case_path(self, case_id):
        return self.directory / f"case_{case_id}.json"

    def completed_ids(self):
        return [case_id for case_id in CASE_IDS if self.case_path(case_id).is_file()]

    def load_case(self, case_id):
        return _read_json(self.case_path(case_id))

    def pending_attempts(self, case_id):
        pending = self.run["pending_case"]
        return pending if pending and pending["case_id"] == case_id else {"case_id": case_id, "attempts_started": 0,
                                                                         "failed_attempts": []}

    # --- state changes (each one persisted immediately) ---------------------------------------
    def mark_running(self, resumed=False):
        self.run["status"] = RUNNING
        self.run["fatal_error"] = None
        if resumed:
            self.run.setdefault("resumed_at", []).append(_timestamp(self.now))
        self._persist()

    def begin_attempt(self, case_id):
        pending = self.pending_attempts(case_id)
        pending["attempts_started"] += 1
        self.run["pending_case"] = pending
        self._persist()
        return pending["attempts_started"]

    def record_failed_attempt(self, case_id, info):
        pending = self.pending_attempts(case_id)
        pending["failed_attempts"].append(info)
        self.run["pending_case"] = pending
        self._persist()

    def save_case(self, case_id, result):
        """Write a completed case exactly once, then clear the pending state and report."""
        path = self.case_path(case_id)
        if path.exists():
            raise CaseAlreadyCompleted(f"{path.name} already exists; a completed case is never overwritten.")
        atomic_write_json(path, result)
        if self.run["pending_case"] and self.run["pending_case"]["case_id"] == case_id:
            self.run["pending_case"] = None
        self._persist()
        return path

    def mark_fatal(self, info):
        self.run["status"] = STOPPED_FATAL
        self.run["fatal_error"] = info
        self._persist()

    def mark_completed(self):
        self.run["status"] = COMPLETED
        self._persist()

    def _persist(self):
        self.run["updated_at"] = _timestamp(self.now)
        atomic_write_json(self.directory / RUN_FILE, self.run)
        atomic_write_json(self.directory / REPORT_FILE, self.build_report())

    # --- report -------------------------------------------------------------------------------
    def build_report(self):
        completed = {case_id: self.load_case(case_id) for case_id in self.completed_ids()}
        counts = {d: {"PASS": 0, "FAIL": 0, "N/A": 0, "NEEDS_REVIEW": 0} for d in DIMENSIONS}
        cases, needs_review, failed = {}, {}, []
        for case_id, result in completed.items():
            dims = result["dimensions"]
            for dimension, value in dims.items():
                counts[dimension][value] += 1
            review = [d for d, v in dims.items() if v == "NEEDS_REVIEW"]
            if review:
                needs_review[case_id] = review
            if result["outcome"] != "completed" or "FAIL" in dims.values():
                failed.append(case_id)
            cases[case_id] = {"name": result["case_name"], "outcome": result["outcome"], "dimensions": dims,
                              "gemini_calls": result["gemini_calls"]["total"],
                              "tool_rounds": result["tool_rounds"], "retry_count": result["retry_count"],
                              "review_flags": result["review_flags"], "completed_at": result["completed_at"]}
        pending = self.run["pending_case"]
        failed_attempt_calls = sum(r["gemini_calls"]["failed_attempts"] for r in completed.values())
        pending_calls = sum(a.get("gemini_calls") or 0 for a in (pending or {}).get("failed_attempts", []))
        completed_calls = {case_id: r["gemini_calls"]["total"] for case_id, r in completed.items()}
        retry_counts = {case_id: r["retry_count"] for case_id, r in completed.items()}
        if pending:
            retry_counts[pending["case_id"]] = _retry_count(pending)
        return {
            "evaluation_version": self.run["evaluation_version"],
            "run_id": self.run["run_id"],
            "status": self.run["status"],
            "started_at": self.run["started_at"],
            "updated_at": self.run["updated_at"],
            "completed_cases": list(completed),
            "remaining_cases": [c for c in CASE_IDS if c not in completed],
            "pending_case": None if not pending else {
                "case_id": pending["case_id"], "attempts_started": pending["attempts_started"],
                "failed_attempts": len(pending["failed_attempts"]),
                "last_failure": pending["failed_attempts"][-1] if pending["failed_attempts"] else None},
            "dimension_counts": counts,
            "needs_review": needs_review,
            "failed_cases": failed,
            "fatal_error": self.run.get("fatal_error"),
            "retry_counts": retry_counts,
            "gemini_calls_per_completed_case": completed_calls,
            "total_gemini_calls_completed_cases": sum(completed_calls.values()),
            "total_gemini_calls_including_failed_attempts":
                sum(completed_calls.values()) + failed_attempt_calls + pending_calls,
            "note": "Gemini calls of an attempt cut off by a crash/power loss are unknown (not counted).",
            "cases": cases,
        }


def _retry_count(pending):
    """Failed attempts plus attempts that were started but never finished (process died)."""
    return max(pending["attempts_started"] - 1, len(pending["failed_attempts"]))
