"""
Wires everything into one LangGraph StateGraph:

    START -> compact --> agent --(no tool_calls)--> END
                          agent --(tool_calls)--> tools --> compact --> agent

`compact` sits in front of EVERY agent call (first turn and every turn
after a tool run) and is a cheap no-op unless the message list has actually
grown past the threshold. This matters: if compaction were only checked
after tool calls, a long tool-free back-and-forth conversation would never
get compacted at all.

The agent node only binds tools that are in state["loaded_tools"], plus the
always-on meta-tools (search_tools, load_tools) and scratchpad tools
(read_file, write_file, list_files) -- those five are cheap enough to keep
resident at all times. Everything else in TOOL_REGISTRY stays invisible to
the model until `load_tools` has been called.
"""
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from tools_task import TOOL_REGISTRY
from tools_meta import search_tools, load_tools
from tools_fs import read_file, write_file, list_files
from compaction import compact_node

ALWAYS_ON_TOOLS = [search_tools, load_tools, read_file, write_file, list_files]

SYSTEM_PROMPT = """You are a production agent with context-management skills.

You do NOT have every tool loaded at once. Call `search_tools` to find tools
by keyword, then `load_tools` with the exact name(s) to make them callable.

For anything long -- big search results, drafts, documents -- call
`write_file` to store it in the scratchpad instead of pasting it into your
reply, then `read_file` later to pull back only the part you need."""


def build_graph(chat_llm, summarizer_llm=None):
    """
    chat_llm: main chat model for the agent's turns (tools bound dynamically
        on every call, based on state["loaded_tools"]).
    summarizer_llm: model used inside compaction to write summaries.
        Defaults to chat_llm. In production this is often a cheaper/faster
        model, since summarization doesn't need your strongest model.
    """
    summarizer_llm = summarizer_llm or chat_llm

    def agent_node(state: AgentState) -> dict:
        loaded_names = state.get("loaded_tools", [])
        active_tools = ALWAYS_ON_TOOLS + [
            TOOL_REGISTRY[name]["tool"] for name in loaded_names if name in TOOL_REGISTRY
        ]
        model_with_tools = chat_llm.bind_tools(active_tools)

        # Prepend the system prompt for this call only -- it is NOT saved
        # back into state, so it doesn't pile up in the message history
        # across turns (and doesn't need to survive compaction either).
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def compact_node_wrapper(state: AgentState) -> dict:
        return compact_node(state, summarizer_llm)

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    # ToolNode needs the actual callables for every tool that could ever be
    # invoked, even ones not currently "loaded" -- loading only controls
    # what the LLM sees in its schema, not what the executor can run.
    all_executable_tools = ALWAYS_ON_TOOLS + [info["tool"] for info in TOOL_REGISTRY.values()]
    tool_node = ToolNode(all_executable_tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("compact", compact_node_wrapper)

    graph.add_edge(START, "compact")
    graph.add_edge("compact", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "compact")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
