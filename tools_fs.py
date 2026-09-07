"""
Scratchpad / virtual-filesystem tools.

Large content (raw search results, long drafts, documents) gets written
here instead of being pasted straight into the message history. The agent
reads back only the slice it actually needs, instead of dragging the whole
blob through every subsequent LLM call.
"""
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


@tool
def write_file(
    path: str,
    content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Write content to the scratchpad under the given path. Use this for
    anything long -- raw tool output, drafts, documents -- instead of
    putting it directly into your reply or leaving it in a tool result.
    """
    scratchpad = dict(state.get("scratchpad", {}))
    scratchpad[path] = content
    return Command(
        update={
            "scratchpad": scratchpad,
            "messages": [
                ToolMessage(
                    content=f"Wrote {len(content)} chars to '{path}'.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def read_file(path: str, state: Annotated[dict, InjectedState]) -> str:
    """Read content previously written to the scratchpad at `path`."""
    scratchpad = state.get("scratchpad", {})
    if path not in scratchpad:
        return f"No file at '{path}'. Known files: {list(scratchpad.keys())}"
    return scratchpad[path]


@tool
def list_files(state: Annotated[dict, InjectedState]) -> str:
    """List all paths currently stored in the scratchpad, with sizes."""
    scratchpad = state.get("scratchpad", {})
    if not scratchpad:
        return "Scratchpad is empty."
    return "\n".join(f"- {p} ({len(c)} chars)" for p, c in scratchpad.items())
