"""
Example 'real' task tools, plus a lightweight registry that pairs each one
with a SHORT one-line description. `search_tools` (in tools_meta.py) only
ever looks at these short descriptions -- never the tools' full schemas --
which is what keeps the always-visible context small.

Swap these stubs for your actual tools (DB calls, external APIs, etc).
The pattern (define tool -> register with a short description) stays the
same no matter how many real tools you have.
"""
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: expression contains disallowed characters."
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (stub implementation)."""
    return f"It is sunny and 24C in {city}."


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol (stub implementation)."""
    return f"{ticker.upper()} is trading at $123.45."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (stub implementation)."""
    return f"Email sent to {to} with subject '{subject}'."


@tool
def translate_text(text: str, target_language: str) -> str:
    """Translate text into a target language (stub implementation)."""
    return f"[{target_language} translation of]: {text}"


# name -> {"tool": <BaseTool>, "short_desc": <str shown to the model by search_tools>}
TOOL_REGISTRY: dict[str, dict] = {
    "calculator": {"tool": calculator, "short_desc": "Evaluate arithmetic expressions."},
    "get_weather": {"tool": get_weather, "short_desc": "Get current weather for a city."},
    "get_stock_price": {
        "tool": get_stock_price,
        "short_desc": "Get latest stock price for a ticker.",
    },
    "send_email": {"tool": send_email, "short_desc": "Send an email to a recipient."},
    "translate_text": {
        "tool": translate_text,
        "short_desc": "Translate text into another language.",
    },
}
