"""
Placeholder tools for the Day 4 sub-agent exercise.

These were originally bare functions with no @tool decorator and no
arguments, which meant `tool.name` raised AttributeError and bind_tools()
rejected them -- any delegation crashed. They are real tools now.

Swap the bodies for actual implementations (a search API, a sandboxed
executor) when the capstone needs them; the registry shape stays the same.
"""
from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """Search the web for `query` and return the findings (stub).

    Args:
        query: What to search for.
    """
    return f"[stub] search results for: {query}"


@tool
def code_exec_tool(code: str) -> str:
    """Execute a snippet of Python and return its output (stub).

    Real implementations MUST run this in a sandbox -- see Day 10.

    Args:
        code: The Python source to execute.
    """
    return f"[stub] executed {len(code)} chars of code"
