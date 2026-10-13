"""Chainlit web chat (V1) for the Airport Investment Intelligence Agent.

Presentation layer only: every user message goes to the existing AgentSession.send();
the Agent loop, Gemini integration, memory, tools and tracing are unchanged.

    browser -> on_message -> this chat's AgentSession (in a worker thread) -> answer -> browser

Run:  chainlit run chat_app.py
"""

import asyncio
import logging

import chainlit as cl

from agent import AgentSession
from chat_support import STARTERS, execution_summary, friendly_error

logger = logging.getLogger("airport_agent.chat")
SESSION_KEY, LOCK_KEY = "agent_session", "agent_lock"


@cl.set_starters
async def starters(user=None, language=None):
    return [cl.Starter(label=label, message=message) for label, message in STARTERS]


@cl.on_chat_start
async def start_chat():
    """One AgentSession (conversation, tool context and data cache) per browser chat."""
    cl.user_session.set(SESSION_KEY, AgentSession())
    cl.user_session.set(LOCK_KEY, asyncio.Lock())


@cl.on_message
async def on_message(message: cl.Message):
    session, lock = cl.user_session.get(SESSION_KEY), cl.user_session.get(LOCK_KEY)
    async with lock:                                   # one turn at a time per chat
        answer = None
        async with cl.Step(name="Working on your question", type="run", show_input=False) as step:
            # AgentSession.send() is synchronous (model + data requests): run it in a worker
            # thread so the server keeps serving other chats.
            turn = asyncio.ensure_future(cl.make_async(session.send)(message.content))
            try:
                answer = await asyncio.shield(turn)
            except asyncio.CancelledError:
                # Stop pressed: the worker thread cannot be interrupted. Keep this chat's lock until
                # the turn has really ended, so a new message never runs concurrently on the session.
                await asyncio.wait([turn])
                step.output = "Stopped."
                raise
            except Exception as exc:                   # caught here: the step must never show str(exc)
                logger.exception("Agent turn failed")
                step.output = "Stopped."
                error_text = friendly_error(exc)
        if answer is None:
            await cl.Message(content=error_text).send()
            return
        await step.remove()                            # progress indicator only: nothing to keep once done

        lines = execution_summary(session.last_trace)
        if lines:
            async with cl.Step(name="Execution summary", type="tool", show_input=False,
                               default_open=False) as summary:
                summary.output = "\n".join(f"- {line}" for line in lines)
        await cl.Message(content=answer).send()
