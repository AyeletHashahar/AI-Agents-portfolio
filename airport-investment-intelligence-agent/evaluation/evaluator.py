"""Deterministic evaluation of one live case (no LLM, no network).

Inputs are what the runner observed: the evaluated turn's execution trace
(AgentSession.last_trace.to_dict()), the committed turn's transcript (function calls and
their results, final text; no thought parts), setup turns and request observations.

Only what can be established mechanically is decided here. Everything about prose quality
- grounding, causal claims, limitation wording, recommendation quality - is NEEDS_REVIEW
for the human reviewer; mechanical hints for that review go to `review_flags`. Required and
forbidden behaviors are checked, never one exact tool trajectory.

Dimension values: PASS / FAIL / N/A for Tool Selection, Tool Arguments, Entity Resolution
and Memory Behavior; PASS / FAIL / NEEDS_REVIEW for Grounding and Task Success.
"""

import re

from agent.memory import MEMORY_TOOL_NAME
from agent_tools import AUTOMATIC_ANALYSIS_LIMIT

PASS, FAIL, NA, REVIEW = "PASS", "FAIL", "N/A", "NEEDS_REVIEW"
DIMENSIONS = ("tool_selection", "tool_arguments", "entity_resolution", "memory_behavior", "grounding",
              "task_success")
RESOLVE, DISCOVER, COMPARE, ANALYZE = "resolve_airports", "discover_airports", "compare_airports", "analyze_airport"
ANALYSIS_TOOLS = (COMPARE, ANALYZE)
SETUP_COMPARISON_TURN = 1           # the comparison is the first setup turn of the memory case


# --- extraction ----------------------------------------------------------------------------------

def tool_calls(trace):
    return list((trace or {}).get("tool_calls") or [])


def tool_results(turn):
    """{call_id: function response dict} from a serialized turn transcript."""
    results = {}
    for content in turn or []:
        for part in content["parts"]:
            if "function_response" in part:
                results[part["function_response"]["id"]] = part["function_response"]["response"]
    return results


def _codes(call):
    args = call.get("arguments") or {}
    if call["name"] == COMPARE:
        values = args.get("airport_codes") or []
    elif call["name"] == ANALYZE:
        values = [args.get("airport_code")]
    else:
        return []
    return [v.strip().upper() for v in values if isinstance(v, str)]


def _found_codes(result):
    return [r["airport"]["code"] for r in (result or {}).get("results") or []
            if r.get("status") == "found" and r.get("airport", {}).get("code")]


def _discovered(result):
    if not result or "columns" not in result:
        return set(), None
    column = result["columns"].index("code")
    return {row[column] for row in result["airports"]}, result.get("candidate_count")


def _memory_codes(result):
    codes = set()
    for match in (result or {}).get("matches") or []:
        for tool in match.get("tools_used") or []:
            codes.update(_codes({"name": tool["name"], "arguments": tool.get("args") or {}}))
    return codes


def _executed(call, results):
    """A business call that actually produced analysis (not an error-only response)."""
    result = results.get(call.get("call_id"))
    return call.get("status") in ("ok", "partial") and bool(result)


def code_provenance(calls, results, prompt=""):
    """For every analysis call: codes not obtained from an EARLIER tool result (resolve,
    discover or memory) and not written by the user as a code."""
    user_codes = set(re.findall(r"\b[A-Z0-9]{3,4}\b", prompt))
    known, unexplained = set(user_codes), []
    for call in calls:
        result = results.get(call.get("call_id"))
        if call["name"] in ANALYSIS_TOOLS:
            missing = sorted(set(_codes(call)) - known)
            if missing:
                unexplained.append({"call_id": call.get("call_id"), "tool": call["name"], "codes": missing})
        if call["name"] == RESOLVE:
            known.update(_found_codes(result))
        elif call["name"] == DISCOVER:
            known.update(_discovered(result)[0])
        elif call["name"] == MEMORY_TOOL_NAME:
            known.update(_memory_codes(result))
    return unexplained


def tool_values(calls, results):
    """Key numbers the tools returned in this turn (for the reviewer and mechanical checks)."""
    values = {}
    for call in calls:
        result = results.get(call.get("call_id")) or {}
        if call["name"] == ANALYZE and result.get("airport"):
            entry = values.setdefault(result["airport"]["code"], {"name": result["airport"]["name"]})
            entry["long_haul_share"] = (result.get("scale_connectivity") or {}).get("signals", {}).get("long_haul_share")
            entry["passenger_cagr"] = (result.get("demand") or {}).get("passenger_cagr")
            entry["capacity_pressure"] = result.get("capacity_pressure")
        elif call["name"] == COMPARE:
            for top in result.get("top_candidates") or []:
                entry = values.setdefault(top["code"], {"name": top.get("name")})
                families = top.get("families") or {}
                entry["investment_score"] = top.get("investment_score")
                entry["rank"] = top.get("rank")
                for family, summary in families.items():
                    entry[f"{family}_score"] = summary.get("score")
                    for signal, detail in (summary.get("signals") or {}).items():
                        entry.setdefault(signal, detail.get("value"))
    growth = {code: v["passenger_cagr"] for code, v in values.items() if v.get("passenger_cagr") is not None}
    if len(growth) >= 2:
        winner = max(growth, key=growth.get)
        values["_highest_passenger_cagr"] = {"code": winner, "name": values[winner].get("name"),
                                             "passenger_cagr": growth[winner]}
    return values


# --- checks ------------------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.items, self.reasons, self.flags = [], {d: [] for d in DIMENSIONS}, []

    def add(self, name, result, detail=None, dimension=None):
        """result: True / False / None (informational)."""
        self.items.append({"name": name, "result": result, "detail": detail})
        if dimension and result is False:
            self.reasons[dimension].append(name if detail is None else f"{name}: {detail}")
        return result


def _common(case, record, checks):
    trace, answer = record.get("trace") or {}, record.get("final_answer") or ""
    checks.add("execution_succeeded", record.get("outcome") == "completed" and trace.get("status") == "success",
               record.get("error"), "task_success")
    checks.add("forced_finalization_used", trace.get("finalization_used"))
    checks.add("llm_calls", trace.get("total_llm_calls"))
    checks.add("tool_rounds", trace.get("tool_rounds"))
    for name in case.expected_names:
        mentioned = name.lower() in answer.lower()
        checks.add(f"canonical_name_in_answer:{name}", mentioned)
        if answer and not mentioned:
            checks.flags.append(f"canonical name not used: {name}")
    if trace.get("finalization_used"):
        checks.flags.append("forced finalization: budget exhausted, answer from partial evidence")
    lowered = answer.lower()
    for term in case.flag_terms:
        if term in lowered:
            checks.flags.append(f"contains '{term}': check it is not presented as observed evidence")


def _memory_unused(calls, checks):
    used = any(c["name"] == MEMORY_TOOL_NAME for c in calls)
    checks.add("memory_not_used", not used, None, "memory_behavior")
    return PASS if not used else FAIL


def _airport_cases(case, record, checks, calls, results):
    names = [c["name"] for c in calls]
    expected = set(case.expected_codes)
    analysis = [c for c in calls if c["name"] in ANALYSIS_TOOLS]
    analysed = {code for c in analysis for code in _codes(c)}
    resolved = {code for c in calls if c["name"] == RESOLVE for code in _found_codes(results.get(c.get("call_id")))}

    used_resolve = checks.add("resolve_airports_used", RESOLVE in names, None, "tool_selection")
    no_discover = checks.add("discover_airports_not_used", DISCOVER not in names, None, "tool_selection")
    compared_together = any(c["name"] == COMPARE and expected <= set(_codes(c)) for c in analysis)
    if case.kind == "investment_comparison":        # an Investment Score needs ONE comparison group
        needed = checks.add("expected_airports_compared_as_one_group", compared_together,
                            [_codes(c) for c in analysis if c["name"] == COMPARE] or None, "tool_selection")
    elif case.kind == "airport_comparison":
        covered = compared_together or \
            expected <= {code for c in analysis if c["name"] == ANALYZE for code in _codes(c)}
        needed = checks.add("all_expected_airports_analysed", covered, sorted(expected - analysed), "tool_selection")
    elif case.kind == "single_airport":
        needed = checks.add("expected_airport_analysed", expected <= analysed, sorted(expected - analysed),
                            "tool_selection")
    else:                                   # analysis optional (grounding trap)
        needed = True
        checks.add("analysis_tool_used", bool(analysis))
    selection = PASS if used_resolve and no_discover and needed else FAIL

    if analysis:
        extra = sorted(analysed - expected)
        ok = checks.add("analysis_codes_are_expected", not extra, extra, "tool_arguments")
        no_large = checks.add("allow_large_group_not_set",
                              not any((c.get("arguments") or {}).get("allow_large_group") for c in analysis),
                              None, "tool_arguments")
        arguments = PASS if ok and no_large else FAIL
    else:
        arguments = NA

    first_resolve = next((i for i, c in enumerate(calls) if c["name"] == RESOLVE), None)
    first_analysis = next((i for i, c in enumerate(calls) if c["name"] in ANALYSIS_TOOLS), None)
    order_ok = checks.add("resolve_before_analysis", first_analysis is None or
                          (first_resolve is not None and first_resolve < first_analysis), None, "entity_resolution")
    resolved_ok = checks.add("expected_codes_resolved", expected <= resolved, sorted(expected - resolved),
                             "entity_resolution")
    unexplained = code_provenance(calls, results, case.prompt)
    provenance_ok = checks.add("codes_come_from_tool_results", not unexplained, unexplained or None,
                               "entity_resolution")
    entity = PASS if order_ok and resolved_ok and provenance_ok else FAIL
    return selection, arguments, entity, _memory_unused(calls, checks)


def _geographic(case, record, checks, calls, results):
    names = [c["name"] for c in calls]
    discover_calls = [c for c in calls if c["name"] == DISCOVER]
    used = checks.add("discover_airports_used", bool(discover_calls), None, "tool_selection")
    checks.add("resolve_airports_used", RESOLVE in names)           # informational: not required
    discovered, count = set(), None
    for call in discover_calls:
        codes, n = _discovered(results.get(call.get("call_id")))
        discovered |= codes
        count = n if count is None or (n is not None and n > count) else count
    checks.add("discovered_candidate_count", count)
    above = count is not None and count > AUTOMATIC_ANALYSIS_LIMIT
    checks.add("candidate_count_above_automatic_limit", above if count is not None else None)

    compares = [c for c in calls if c["name"] == COMPARE]
    executed = [c for c in compares if _executed(c, results)]
    bypass = [c.get("call_id") for c in compares if (c.get("arguments") or {}).get("allow_large_group")]
    no_bypass = checks.add("allow_large_group_not_set", not bypass, bypass or None, "tool_arguments")
    groups = [set(_codes(c)) for c in executed]
    split = any(a.isdisjoint(b) for i, a in enumerate(groups) for b in groups[i + 1:])
    no_split = checks.add("no_split_comparison_groups", not split, None, "tool_arguments")
    outside = sorted({code for g in groups for code in g} - discovered)
    in_discovery = checks.add("compared_codes_were_discovered", not outside, outside or None, "tool_arguments")

    args_ok = True
    for call in discover_calls:
        args = call.get("arguments") or {}
        states = {s.strip().upper() for s in args.get("states") or [] if isinstance(s, str)}
        right = (str(args.get("country", "")).upper() == "US" and states == set(case.discover_states))
        args_ok = checks.add("discover_arguments", right, {"country": args.get("country"), "states": sorted(states)},
                             "tool_arguments") and args_ok
    arguments = PASS if discover_calls and args_ok and no_bypass and no_split and in_discovery else \
        (FAIL if calls else NA)

    if count is None:
        selection, guardrail_ok = (PASS if used else FAIL), None
        if used:
            checks.add("discovery_result_available", False)
            checks.flags.append("discover_airports returned no candidate list (see transcript)")
    elif above:
        guardrail_ok = checks.add("large_group_guardrail_respected", not executed and not bypass,
                                  [c.get("call_id") for c in executed] or None, "tool_selection")
        selection = PASS if used and guardrail_ok else FAIL
    else:
        guardrail_ok = True
        compared = checks.add("discovered_group_compared", bool(executed), None, "tool_selection")
        selection = PASS if used and compared else FAIL
    if case.id == "07" and count is not None and not above:
        checks.flags.append("premise not met: New England group was within the automatic limit")
    return selection, arguments, NA, _memory_unused(calls, checks), guardrail_ok


def _memory_cases(case, record, checks, calls, results):
    memory = [c for c in calls if c["name"] == MEMORY_TOOL_NAME]
    used = checks.add("memory_used", bool(memory), None, "memory_behavior")
    matched = {t for c in memory for t in (c.get("memory_matched_turns") or [])}
    retrieved = checks.add("comparison_turn_retrieved", SETUP_COMPARISON_TURN in matched, sorted(matched),
                           "memory_behavior")
    queries = [(c.get("arguments") or {}).get("query", "") for c in memory]
    checks.add("memory_queries", queries or None)
    requests = record.get("evaluated_requests") or []
    hidden = checks.add("old_comparison_not_in_initial_context",
                        bool(requests) and not requests[0].get("contains_setup_turn_1"), None, "memory_behavior")
    memory_dim = PASS if used and retrieved and hidden else FAIL

    expected = set(case.expected_codes)
    analysis = [c for c in calls if c["name"] in ANALYSIS_TOOLS]
    unexplained = code_provenance(calls, results, case.prompt)
    no_discover = checks.add("discover_airports_not_used", all(c["name"] != DISCOVER for c in calls), None,
                             "tool_selection")
    checks.add("business_tool_used", bool(analysis))          # allowed, not required
    selection = PASS if used and no_discover else FAIL
    query_ok = checks.add("memory_query_non_empty", bool(queries) and all(q.strip() for q in queries), None,
                          "tool_arguments")
    codes_ok = True
    if analysis:
        extra = sorted({code for c in analysis for code in _codes(c)} - expected)
        codes_ok = checks.add("analysis_codes_are_expected", not extra, extra or None, "tool_arguments")
    arguments = PASS if query_ok and codes_ok else (FAIL if memory or analysis else NA)
    entity = NA if not analysis else (PASS if checks.add("codes_come_from_tool_results", not unexplained,
                                                         unexplained or None, "entity_resolution") else FAIL)
    return selection, arguments, entity, memory_dim


def _no_tools(case, record, checks, calls, results):
    """Conceptual question answerable from the Agent's instructions: no tool of any kind."""
    business = [c["name"] for c in calls if c["name"] != MEMORY_TOOL_NAME]
    no_business = checks.add("no_business_tools_used", not business, business or None, "tool_selection")
    memory = _memory_unused(calls, checks)
    selection = PASS if no_business and memory == PASS else FAIL
    if memory == FAIL:
        checks.reasons["tool_selection"].append("memory_not_used")
    answer = record.get("final_answer") or ""
    codes = sorted(set(re.findall(r"\b[A-Z]{3}\b", answer)) - {"CAGR", "KPI", "BTS", "USA", "FAA"})
    if codes:
        checks.flags.append(f"answer mentions code-like tokens {codes}: check no airport-specific facts were invented")
    if re.search(r"\d", answer):
        checks.flags.append("answer contains digits: check none is an invented airport-specific number")
    return selection, NA, NA, memory


def setup_checks(case, setup):
    """Validity of the memory cases' real setup turns: all completed, and turn 1 compared the
    three expected airports. Returns (valid, list of check dicts)."""
    checks = Checks()
    checks.add("setup_turns_completed", len(setup) == len(case.setup) and
               all(s.get("outcome") == "completed" for s in setup))
    first = setup[0] if setup else {}
    calls, results = tool_calls(first.get("trace")), tool_results(first.get("turn"))
    compared = any(c["name"] == COMPARE and set(_codes(c)) == set(case.expected_codes) and _executed(c, results)
                   for c in calls)
    checks.add("setup_turn_1_compared_expected_airports", compared, [_codes(c) for c in calls if c["name"] == COMPARE])
    return all(item["result"] for item in checks.items), checks.items


def evaluate_case(case, record):
    """Return {"checks", "dimensions", "dimension_reasons", "review_flags", "tool_values"}."""
    checks = Checks()
    if record.get("outcome") == "setup_invalid":
        checks.add("setup_valid", False, None, "task_success")
        dims = {d: NA for d in DIMENSIONS[:4]}
        dims.update(grounding=REVIEW, task_success=FAIL)
        checks.flags.append("setup invalid: the evaluated turn was not run (see setup)")
        return _result(checks, dims, {})

    calls, results = tool_calls(record.get("trace")), tool_results(record.get("turn"))
    _common(case, record, checks)
    succeeded = record.get("outcome") == "completed"
    guardrail_ok = None
    if case.kind in ("single_airport", "single_airport_optional_analysis", "airport_comparison",
                     "investment_comparison"):
        selection, arguments, entity, memory = _airport_cases(case, record, checks, calls, results)
    elif case.kind == "geographic":
        selection, arguments, entity, memory, guardrail_ok = _geographic(case, record, checks, calls, results)
    elif case.kind == "memory_recall":
        selection, arguments, entity, memory = _memory_cases(case, record, checks, calls, results)
    elif case.kind == "no_tools":
        selection, arguments, entity, memory = _no_tools(case, record, checks, calls, results)
    else:
        raise ValueError(f"Unknown case kind: {case.kind}")

    values = tool_values(calls, results)
    if record.get("setup"):              # memory cases: what the old comparison returned
        first = record["setup"][0]
        values["_setup_turn_1"] = tool_values(tool_calls(first.get("trace")), tool_results(first.get("turn")))
    answer = record.get("final_answer") or ""
    _value_hints(case, values, answer, checks)
    task = REVIEW
    if not succeeded:
        task = FAIL
    elif guardrail_ok is False:
        task = FAIL
        checks.reasons["task_success"].append("large-group guardrail bypassed")
    dims = {"tool_selection": selection, "tool_arguments": arguments, "entity_resolution": entity,
            "memory_behavior": memory, "grounding": REVIEW, "task_success": task}
    if not answer:
        checks.flags.append("no final answer to review")
    return _result(checks, dims, values)


def _value_hints(case, values, answer, checks):
    """Mechanical hints for the reviewer: are the tool numbers / winner mentioned?"""
    if case.id == "01":
        share = next((v.get("long_haul_share") for code, v in values.items()
                      if code in case.expected_codes and v.get("long_haul_share") is not None), None)
        if share is not None:
            forms = {f"{share * 100:.0f}", f"{share * 100:.1f}", f"{share * 100:.2f}"}
            checks.add("long_haul_share_cited", any(form in answer for form in forms), sorted(forms))
            if not any(form in answer for form in forms):
                checks.flags.append("long-haul share from the tool not found verbatim in the answer")
    source = values.get("_setup_turn_1", {}) if case.id == "09" else values
    winner = source.get("_highest_passenger_cagr")
    if case.kind == "investment_comparison":
        ranked = sorted((v for k, v in values.items() if not k.startswith("_") and v.get("rank") is not None),
                        key=lambda v: v["rank"])
        if ranked:
            top = ranked[0]
            checks.add("rank_1_airport_mentioned", top["name"].lower() in answer.lower() if top.get("name") else None,
                       {"name": top.get("name"), "investment_score": top.get("investment_score")})
    if winner and case.id in ("03", "09"):
        mentioned = winner["code"] in answer or (winner["name"] or "").lower() in answer.lower()
        checks.add("highest_cagr_airport_mentioned", mentioned, winner)


def _result(checks, dims, values):
    return {"checks": checks.items, "dimensions": dims,
            "dimension_reasons": {d: r for d, r in checks.reasons.items() if r},
            "review_flags": checks.flags, "tool_values": values}
