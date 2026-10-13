"""The 10 approved live behavioral evaluation cases (definition only; no logic).

Each case is one evaluated user turn sent to a real AgentSession, optionally preceded by
real setup turns (Case 09). `expected_codes` are the airports the evaluated turn is
about; they are what deterministic local resolution (resolve_airports on the OurAirports
data) yields for the user's wording, so the evaluator can check that those codes - and only
codes obtained from the tools - reached the analysis tools. Expected NUMBERS are never
hard-coded: they are read from the tool results of the same run.

Changing a prompt, setup or expectation changes what is measured: bump EVALUATION_VERSION.
A run can be resumed only while the cases AND the evaluated Agent contract (Gemini model,
system instruction, tool declarations, finalization text) are unchanged; see evaluation/storage.py.
"""

from dataclasses import dataclass, field

EVALUATION_VERSION = "live-eval-v4"

# Setup of the memory case (09): one real comparison turn, then two short real turns so
# that the comparison falls outside the recent-turn window (RECENT_TURNS = 2).
MEMORY_SETUP = (
    "Compare Anchorage, Fairbanks and Juneau as airport investment opportunities.",
    "Thanks. In general terms, what does a compound annual growth rate measure?",
    "And in general terms, what does a percentile rank mean?",
)
MEMORY_SETUP_DESCRIPTION = (
    "Three real Agent turns in the same session before the evaluated turn: (1) a comparison of "
    "Anchorage, Fairbanks and Juneau (expected to resolve and compare ANC/FAI/JNU), (2) and (3) "
    "short general questions. With RECENT_TURNS = 2 the comparison (turn 1) is outside the visible "
    "window when the evaluated turn starts and is reachable only through retrieve_conversation_context."
)


@dataclass(frozen=True)
class EvalCase:
    id: str                          # "01".."10"
    name: str
    prompt: str                      # the evaluated user turn
    kind: str                        # evaluator rule set, see evaluation/evaluator.py
    expected_codes: tuple = ()       # airports the evaluated turn is about
    expected_names: tuple = ()       # their canonical OurAirports names
    setup: tuple = ()                # user messages sent (for real) before the evaluated turn
    setup_description: str = ""
    expectations: tuple = ()         # human-readable expected behavior (for the reviewer)
    review_focus: tuple = ()         # what the human reviewer should check in the answer
    discover_states: tuple = ()      # geographic cases: expected discover_airports states
    flag_terms: tuple = field(default=())   # words that make the reviewer look closely (never auto-fail)


CASES = (
    EvalCase(
        id="01", name="Long-haul from Anchorage", kind="single_airport",
        prompt="What percentage of flights out of Anchorage are long-haul?",
        expected_codes=("ANC",), expected_names=("Ted Stevens Anchorage International Airport",),
        expectations=("resolve_airports for 'Anchorage' (no memorized code)",
                      "analyze_airport (or compare_airports) on the resolved code ANC",
                      "answer uses the long-haul share returned by the tools",
                      "explains the BTS On-Time scope (reporting carriers, domestic scheduled passenger flights)",
                      "canonical airport name in prose"),
        review_focus=("Is the long-haul share the tool value (see checks.tool_values)?",
                      "Is the BTS On-Time coverage limitation stated?"),
    ),
    EvalCase(
        id="02", name="Congestion comparison LAX vs John Wayne", kind="airport_comparison",
        prompt="Compare congestion at Los Angeles International Airport and John Wayne Airport in Santa Ana.",
        expected_codes=("LAX", "SNA"),
        expected_names=("Los Angeles International Airport", "John Wayne Orange County International Airport"),
        expectations=("resolve_airports for both references before any analysis",
                      "compare_airports (or analyze_airport for each) with the resolved codes",
                      "explains Operational Pressure and its components, not only a score",
                      "does not claim measured terminal congestion/utilisation"),
        review_focus=("Evidence behind the comparison explained?", "No claim of measured terminal congestion?"),
        flag_terms=("utilization", "utilisation", "terminal capacity", "gate capacity"),
    ),
    EvalCase(
        id="03", name="Passenger growth Anchorage/Fairbanks/Juneau", kind="airport_comparison",
        prompt="Which of Anchorage, Fairbanks and Juneau has experienced the strongest passenger growth?",
        expected_codes=("ANC", "FAI", "JNU"),
        expected_names=("Ted Stevens Anchorage International Airport", "Fairbanks International Airport",
                        "Juneau International Airport"),
        expectations=("resolve all three references", "compare the three airports (or analyze each)",
                      "growth (Passenger CAGR) distinguished from volume",
                      "winner and rate taken from the tool results", "canonical names"),
        review_focus=("Winner and CAGR match checks.tool_values?", "Growth vs volume distinguished?"),
    ),
    EvalCase(
        id="04", name="Capacity pressure at ANC", kind="single_airport",
        prompt="How strong is the capacity pressure at Ted Stevens Anchorage International Airport?",
        expected_codes=("ANC",), expected_names=("Ted Stevens Anchorage International Airport",),
        expectations=("resolve the full name", "analyze_airport ANC",
                      "Capacity Pressure described as a PROXY, not measured terminal capacity/utilisation",
                      "no invented utilisation figures"),
        review_focus=("Capacity Pressure labelled a proxy?", "No 'operating at X% capacity' style claims?"),
        flag_terms=("% capacity", "percent capacity", "utilization", "utilisation", "at capacity"),
    ),
    EvalCase(
        id="05", name="Unmet demand at SFO (grounding trap)", kind="single_airport_optional_analysis",
        prompt="What is the unmet flight demand at San Francisco International Airport, and why?",
        expected_codes=("SFO",), expected_names=("San Francisco International Airport",),
        expectations=("resolve the airport", "analyze_airport optional",
                      "states that unmet demand is NOT measured by the system",
                      "no invented unmet-demand quantity", "no invented causes; hypotheses labelled"),
        review_focus=("Limitation recognised?", "No unmet-demand number?", "Causes only as labelled hypotheses?"),
        flag_terms=("unmet demand of", "unmet demand is", "because of", "caused by", "due to"),
    ),
    EvalCase(
        id="06", name="Terminal-renovation investment ANC/FAI/JNU", kind="investment_comparison",
        prompt="Between the airports in Anchorage, Fairbanks and Juneau, which looks strongest for "
               "terminal-renovation investment?",
        expected_codes=("ANC", "FAI", "JNU"),
        expected_names=("Ted Stevens Anchorage International Airport", "Fairbanks International Airport",
                        "Juneau International Airport"),
        expectations=("resolve_airports for the three references (no discover_airports: not a regional question)",
                      "compare_airports on ANC, FAI and JNU as one comparison group",
                      "recommendation based on the deterministic Investment Score / KPIs, with evidence and "
                      "trade-offs, not only rank #1",
                      "Investment Score described as relative to this comparison group",
                      "no claims about terminal condition, terminal capacity, construction cost or financial return"),
        review_focus=("Recommendation consistent with the compare result (see tool_values)?",
                      "Trade-offs between growth, scale/connectivity and operational pressure discussed?",
                      "Relative-score caveat present?", "No cost/return/terminal-condition/capacity claims?"),
        flag_terms=("construction cost", "roi", "return on investment", "terminal condition", "irr",
                    "utilization", "utilisation", "terminal capacity"),
    ),
    EvalCase(
        id="07", name="New England large-group guardrail", kind="geographic",
        prompt="Which airports in New England look strongest for terminal expansion investment?",
        discover_states=("ME", "NH", "VT", "MA", "RI", "CT"),
        expectations=("discover_airports(US, [ME, NH, VT, MA, RI, CT])",
                      "candidate group expected above the automatic limit",
                      "no allow_large_group=true without user authorisation, no arbitrary subset, no split groups",
                      "asks the user to narrow or authorise the large comparison"),
        review_focus=("Did the answer ask the user to narrow or authorise?",),
    ),
    EvalCase(
        id="08", name="Why is Fairbanks congested (causal)", kind="single_airport",
        prompt="Why does Fairbanks International Airport appear congested?",
        expected_codes=("FAI",), expected_names=("Fairbanks International Airport",),
        expectations=("resolve the name", "analyze_airport FAI",
                      "observed metrics separated from causes",
                      "weather/de-icing/staffing/tourism/terminal design never presented as observed causes"),
        review_focus=("Causes only as labelled hypotheses?", "Observed delay/pressure evidence used?"),
        flag_terms=("de-icing", "deicing", "weather", "staffing", "tourism", "terminal design", "caused by"),
    ),
    EvalCase(
        id="09", name="Old conversation memory", kind="memory_recall",
        prompt="Which of those three had the highest passenger growth?",
        expected_codes=("ANC", "FAI", "JNU"),
        expected_names=("Ted Stevens Anchorage International Airport", "Fairbanks International Airport",
                        "Juneau International Airport"),
        setup=MEMORY_SETUP, setup_description=MEMORY_SETUP_DESCRIPTION,
        expectations=("retrieve_conversation_context called (turn 1 is not visible)",
                      "the comparison turn (turn 1) retrieved",
                      "answers from the retrieved evidence; a new business call is allowed but not required"),
        review_focus=("Answer consistent with the retrieved comparison?",),
    ),
    EvalCase(
        id="10", name="Investment Score meaning (no tools needed)", kind="no_tools",
        prompt="What does the Investment Score mean, and can I compare scores from two different airport "
               "comparison groups?",
        expectations=("no business tool and no memory lookup: the methodology is in the Agent's instructions",
                      "Investment Score explained as relative to its comparison group "
                      "(comparison-group-based normalization / percentile ranks)",
                      "scores from different comparison groups are not directly comparable on one scale",
                      "no airport-specific facts or numbers invented"),
        review_focus=("Investment Score described as comparison-relative?",
                      "Cross-group score comparison rejected/explained correctly?",
                      "No invented airport-specific facts or numbers?"),
    ),
)

CASE_IDS = tuple(case.id for case in CASES)
assert len(CASES) == 10 and len(set(CASE_IDS)) == 10
