"""On-demand conversation memory: lets Gemini look up EARLIER turns of the conversation
that are not in its visible (recent) context.

Separate from the business tools and from AgentToolContext: it reads only the session's
committed conversation turns. Gemini sees it as one native function,
retrieve_conversation_context(query); agent/loop.py routes that name here and every other
name to the business-tool dispatcher.

Retrieval (V1, deterministic, LEXICAL - not semantic search):
- candidates: committed turns OLDER than the recent turns already sent to the model;
- each turn is reduced to plain data: turn number, user message, final answer, business
  tools used (name + arguments). No Gemini Content is returned, so retrieved history is
  memory evidence and can never break the current function-call ordering;
- score = number of distinct query terms found in the turn (lower-cased words; words of 5+
  letters compared by their first 5 letters, so "comparison" ~ "compare"; common words
  ignored); ties go to the more recent turn;
- no matching term: no turns are returned (never an unrelated "closest" turn);
- hard limits: at most MAX_MEMORY_RESULTS turns, and long texts are truncated.

Language (V1): terms are English letters and digits only. Gemini is told to write the query as
concise ENGLISH search terms even when the user writes in another language (e.g. Hebrew
"מה אמרנו קודם על העיכובים ב-FAI?" -> query "FAI delays"); no translation step or extra LLM call.
Known limitation: an earlier turn written only in a non-English script, with no English terms or
identifiers (airport codes, numbers), cannot be found.
"""

import json
import re
from collections.abc import Mapping

from google.genai import types

from agent.tools import INVALID_TOOL_ARGUMENTS
from agent_tools.common import error_response, issue
from config import MAX_MEMORY_RESULTS

MEMORY_TOOL_NAME = "retrieve_conversation_context"
# MAX_MEMORY_RESULTS (earlier turns returned per retrieval): see config.py.
MAX_USER_MESSAGE_CHARS = 500
MAX_ANSWER_CHARS = 2000

MEMORY_DECLARATION = types.FunctionDeclaration(
    name=MEMORY_TOOL_NAME,
    description=(
        "Look up EARLIER parts of this conversation that are not in your visible context. Use it only "
        "when the user's request depends on earlier conversation you cannot see (e.g. 'the answer you "
        "gave earlier about FAI delays', 'the second airport from that earlier comparison'). Describe "
        "what you need in `query` as concise ENGLISH search terms (airport codes, metric names, topics), "
        "even if the user writes in another language - e.g. 'FAI delays' for a Hebrew question about FAI "
        "delays. Matching is by shared words; if nothing matches, no turns are returned. Returns at most "
        f"{MAX_MEMORY_RESULTS} matching earlier turns (user message, your answer, tools used). It is a record of the conversation, not new airport data: "
        "use the airport tools for current figures."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Concise English search terms for the "
                                                                  "earlier information you need, e.g. 'ANC FAI JNU "
                                                                  "comparison' or 'FAI delays'."}},
        "required": ["query"],
    },
)

_STOPWORDS = frozenset("""
a an and are as at be before but by did do does earlier for from gave give had has have i in is it its me
my of on or our previous prior said that the their them these this those to us was we were what when which
who with you your answer answers conversation discussed told about last turn turns
tell say says mention mentioned talk talked recall remember again ago back
""".split())


def _terms(text):
    """Normalized distinct terms: lower-case words, stopwords removed, 5+ letter words cut to
    their first 5 letters (crude stemming: 'comparison'/'compare', 'delays'/'delay')."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] if len(w) >= 5 else w for w in words if w not in _STOPWORDS}


def _visible_text(content):
    return "".join(p.text for p in (content.parts or []) if p.text and not p.thought).strip()


def _truncate(text, limit):
    return (text[:limit] + " ...[truncated]", True) if len(text) > limit else (text, False)


def summarize_turn(turn, number):
    """Plain-data record of one completed turn (tuple of Gemini Content: user message first,
    the model's final answer last). Memory lookups themselves are not listed as tools."""
    user_message = _visible_text(turn[0])
    answer = _visible_text(turn[-1]) if turn[-1].role == "model" else ""
    tools = [{"name": p.function_call.name, "args": dict(p.function_call.args or {})}
             for content in turn if content.role == "model"
             for p in (content.parts or []) if p.function_call and p.function_call.name != MEMORY_TOOL_NAME]
    return {"turn": number, "user_message": user_message, "assistant_answer": answer, "tools_used": tools}


class ConversationMemory:
    """Read-only view of the committed turns that are NOT in the model's visible context."""

    def __init__(self, turns=(), visible_turns=0):
        """turns: all committed completed turns (oldest first); the last `visible_turns` of
        them are already in the model context and are not searched."""
        self._older = tuple(turns[:max(len(turns) - visible_turns, 0)])
        self._current_turn = len(turns) + 1

    @property
    def searchable_turns(self):
        return len(self._older)

    def retrieve(self, arguments):
        """Answer a retrieve_conversation_context call (LLM-generated `arguments`)."""
        if not isinstance(arguments, Mapping) or set(arguments) != {"query"} \
                or not isinstance(arguments["query"], str) or not arguments["query"].strip():
            return error_response(MEMORY_TOOL_NAME, issue(
                INVALID_TOOL_ARGUMENTS, "Expected exactly one non-empty string argument: query."))
        query = arguments["query"].strip()
        base = {"tool": MEMORY_TOOL_NAME, "status": "ok", "errors": [], "warnings": [],
                "query": _truncate(query, MAX_USER_MESSAGE_CHARS)[0],
                "searchable_earlier_turns": len(self._older), "max_results": MAX_MEMORY_RESULTS}
        if not self._older:
            return {**base, "matches": [],
                    "note": "No earlier conversation exists beyond what is already visible."}

        query_terms = _terms(query)
        scored = []
        for index, turn in enumerate(self._older):
            record = summarize_turn(turn, index + 1)
            tools = [[tool["name"], *tool["args"].values()] for tool in record["tools_used"]]
            text = " ".join([record["user_message"], record["assistant_answer"], json.dumps(tools, default=str)])
            matched = sorted(query_terms & _terms(text))
            scored.append((len(matched), index, record, matched))

        hits = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], -s[1]))[:MAX_MEMORY_RESULTS]
        if not hits:
            return {**base, "matches": [],
                    "note": "No earlier turn shares any search term with the query. Do not guess what was "
                            "said; retry with different concise English terms, or tell the user."}
        matches = [self._match(record, {"type": "lexical", "matched_terms": matched})
                   for _, _, record, matched in hits]
        return {**base, "matches": matches,
                "note": "Earlier conversation records (memory evidence), best lexical matches first."}

    def _match(self, record, match):
        user_message, user_cut = _truncate(record["user_message"], MAX_USER_MESSAGE_CHARS)
        answer, answer_cut = _truncate(record["assistant_answer"], MAX_ANSWER_CHARS)
        return {**record, "turns_ago": self._current_turn - record["turn"], "user_message": user_message,
                "assistant_answer": answer, "truncated": user_cut or answer_cut, "match": match}
