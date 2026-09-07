"""
Compaction: keeps the message list from growing without bound by
summarizing older messages once the conversation is getting large, and
deleting them from state via RemoveMessage.
"""
import uuid
from langchain_core.messages import (
    BaseMessage,
    RemoveMessage,
    SystemMessage,
    HumanMessage,
)

# Keep the last N raw messages untouched; summarize everything before that.
KEEP_LAST_N_MESSAGES = 6

# Rough token budget before compaction triggers. In production, derive this
# from the actual model's context window (e.g. 128_000) and set the
# threshold at ~70-80% of it, minus room for the next response.
TOKEN_THRESHOLD = 3000


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """
    Cheap ~4-chars-per-token estimate. Swap in tiktoken for an exact count
    against a specific model, e.g.:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return sum(len(enc.encode(str(m.content))) for m in messages)
    """
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4


def needs_compaction(messages: list[BaseMessage]) -> bool:
    return (
        estimate_tokens(messages) > TOKEN_THRESHOLD
        and len(messages) > KEEP_LAST_N_MESSAGES
    )


SUMMARY_PROMPT = """You are compacting an AI agent's conversation history to save space.

Summarize the messages below into a compact but COMPLETE summary. You must preserve:
- Any concrete facts, numbers, file paths, IDs, or names mentioned
- Decisions that were made and why
- Tasks that are still open / not yet completed
- The current state of any ongoing work

Do not preserve small talk or tool-call plumbing -- only what's needed to continue the task.

Messages to summarize:
{transcript}

Write the summary now, as a few dense paragraphs or a bullet list."""


def _messages_to_transcript(messages: list[BaseMessage]) -> str:
    return "\n".join(f"[{m.type}] {m.content}" for m in messages)


def compact_node(state: dict, llm) -> dict:
    """
    Graph node body. `llm` is any plain LangChain chat model (no tools
    bound -- this call is text-in, text-out summarization).

    Returns {} (no-op) if compaction isn't needed yet.
    """
    messages = state["messages"]

    if not needs_compaction(messages):
        return {}

    old_messages = messages[:-KEEP_LAST_N_MESSAGES]
    recent_messages = messages[-KEEP_LAST_N_MESSAGES:]

    transcript = _messages_to_transcript(old_messages)
    summary_response = llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(transcript=transcript))])

    summary_message = SystemMessage(
        content=f"[Summary of earlier conversation]\n{summary_response.content}"
    )

    # Remove EVERY existing message (old + recent), then re-add the summary
    # followed by fresh copies of the recent ones with brand-new ids. This
    # guarantees the final order is correct -- summary first, recent turns
    # after -- rather than relying on where the reducer happens to append
    # a message whose id it hasn't seen before.
    removals = [RemoveMessage(id=m.id) for m in messages if m.id is not None]
    fresh_recent = [m.model_copy(update={"id": str(uuid.uuid4())}) for m in recent_messages]

    return {"messages": removals + [summary_message] + fresh_recent}
