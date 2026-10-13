"""AgentSession: one in-memory, multi-turn conversation with the agent.

Three kinds of state, kept separate:
- STORED conversation: every committed completed turn, kept in full as the original Gemini
  Content (user message, the model's function-call turns, user-role function responses,
  final answer). Nothing is ever dropped from it.
- MODEL context: what one request actually sends to Gemini - only the last `recent_turns`
  completed turns plus the new message. Older turns reach the model only when it asks for
  them through retrieve_conversation_context (agent/memory.py), as bounded plain-data records.
- AgentToolContext: the business tools' clients and raw-data cache. One per session, reused
  across turns, never sent to Gemini and never part of the conversation.

Turns are atomic: a turn runs on copies, and is appended to the stored conversation only
when it finishes successfully. If it raises, the session is unchanged. Memory lookups only
read a snapshot of committed turns. The system instruction travels in Gemini's native config.

Execution trace (agent/trace.py) is separate from conversation state: `last_trace` describes
the most recently ATTEMPTED turn - successful or failed - while only successful turns are
committed to the conversation.
"""

import agent.loop as loop
from agent.trace import TurnRecorder
from agent_tools import AgentToolContext


class AgentSession:
    def __init__(self, *, client=None, context=None, model=loop.MODEL,
                 max_tool_rounds=loop.MAX_TOOL_ROUNDS, max_llm_calls=loop.MAX_LLM_CALLS,
                 recent_turns=loop.RECENT_TURNS):
        """client:       google-genai Client (or compatible); created on the first send() if
                         None, then reused for the whole session.
        context:      AgentToolContext for this session; a new one is created if None.
        recent_turns: completed prior turns sent with each new message (older ones are
                      available to the model on demand).
        Budgets apply per user turn (each send() starts fresh)."""
        if context is not None and not isinstance(context, AgentToolContext):
            raise TypeError("context must be an AgentToolContext.")
        self.context = context if context is not None else AgentToolContext()
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.max_llm_calls = max_llm_calls
        self.recent_turns = recent_turns
        self._client = client
        self._turns = []            # committed completed turns; each a tuple of Gemini Content
        self._last_trace = None

    @property
    def history(self):
        """The full stored conversation, flattened (read-only tuple of Gemini Content)."""
        return tuple(content for turn in self._turns for content in turn)

    @property
    def last_trace(self):
        """agent.trace.TurnTrace of the most recent send() - successful or failed - or None
        before the first one. Frozen; each send() replaces it (no trace history is kept)."""
        return self._last_trace

    def send(self, message):
        """Send one user message; return the agent's final answer text.
        Exceptions propagate; on failure the stored conversation is left unchanged, and
        last_trace records the failed attempt."""
        recorder = TurnRecorder()
        try:
            if self._client is None:
                self._client = loop.create_gemini_client()
            text, turn = loop.run_conversation_turn(
                self._turns, message, self.context, client=self._client, model=self.model,
                max_tool_rounds=self.max_tool_rounds, max_llm_calls=self.max_llm_calls,
                recent_turns=self.recent_turns, trace=recorder)
        except Exception as exc:
            self._last_trace = recorder.finish(exc)
            raise
        self._turns = self._turns + [turn]      # commit only after the whole turn succeeded
        self._last_trace = recorder.finish()
        return text
