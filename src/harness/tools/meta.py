"""
Meta-tools for lazy tool loading.

These two tools are ALWAYS bound to the model -- their own schemas are tiny
(a query string / a list of names), so keeping them resident costs almost
nothing. Every OTHER tool's full schema stays invisible to the model until
`load_tools` has been called for it.
"""
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from harness.tools.registry import TOOL_REGISTRY


@tool
def search_tools(query: str) -> str:
    """
    Search available tools by keyword. Returns short one-line descriptions
    of matching tools. Call this first if you're not sure of the exact
    tool name you need, then call load_tools with the name(s) you want.
    """
    query_lower = query.lower()
    matches = [
        f"- {name}: {info['short_desc']}"
        for name, info in TOOL_REGISTRY.items()
        if query_lower in name.lower() or query_lower in info["short_desc"].lower()
    ]
    if not matches:
        all_tools = [f"- {n}: {i['short_desc']}" for n, i in TOOL_REGISTRY.items()]
        return "No direct match. All available tools:\n" + "\n".join(all_tools)
    return "\n".join(matches)


@tool
def load_tools(
    names: list[str],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Load one or more tools by exact name so their full schema becomes
    available on the NEXT model turn. Use search_tools first if you're
    not sure of the exact name.
    """
    valid = [n for n in names if n in TOOL_REGISTRY]
    invalid = [n for n in names if n not in TOOL_REGISTRY]

    current = set(state.get("loaded_tools", []))
    current.update(valid)

    msg = f"Loaded tools: {valid}." if valid else "No valid tools loaded."
    if invalid:
        msg += f" Unknown tool names (ignored): {invalid}."

    return Command(
        update={
            "loaded_tools": sorted(current),
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )
