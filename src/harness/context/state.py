"""
Shared state definition for the context-engineered agent.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full message history. The `add_messages` reducer knows how to append
    # new messages AND how to delete messages when a RemoveMessage is
    # returned by a node (that's what compaction uses).
    messages: Annotated[list[BaseMessage], add_messages]

    # Virtual filesystem / scratchpad. Large blobs (search results, drafts,
    # long documents) live here instead of in `messages`, so they don't get
    # re-paid-for as tokens on every subsequent LLM call.
    scratchpad: dict[str, str]

    # Names of "real" task tools currently bound to the model. Starts empty
    # (or small) and only grows when the agent explicitly calls `load_tools`.
    loaded_tools: list[str]
