import json
import re
from datetime import datetime
from typing import Any

try:
    import numexpr as _numexpr
    _HAS_NUMEXPR = True
except ImportError:
    _numexpr = None  # type: ignore[assignment]
    _HAS_NUMEXPR = False

try:
    from duckduckgo_search import DDGS as _DDGS
    _HAS_DDGS = True
except ImportError:
    _DDGS = None  # type: ignore[assignment,misc]
    _HAS_DDGS = False


# Tool definitions in OpenAI function-calling format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe math expression, e.g. '2 ** 10 + sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (1-5)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def calculate(expression: str) -> str:
    if not _HAS_NUMEXPR:
        # Fallback: restrict to safe subset via regex then eval
        if re.search(r"[^0-9\s\+\-\*\/\(\)\.\%\^]", expression):
            return "Error: unsafe expression"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"
    try:
        result = _numexpr.evaluate(expression)  # type: ignore[union-attr]
        return str(float(result))
    except Exception as e:
        return f"Error: {e}"


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def web_search(query: str, max_results: int = 3) -> str:
    if not _HAS_DDGS:
        return "Web search unavailable (duckduckgo_search not installed)."
    try:
        results = []
        with _DDGS() as ddgs:  # type: ignore[operator]
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"**{r['title']}**\n{r['body']}\n{r['href']}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def dispatch_tool(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if name == "calculate":
        return calculate(arguments.get("expression", ""))
    if name == "get_datetime":
        return get_datetime()
    if name == "web_search":
        return web_search(arguments.get("query", ""), arguments.get("max_results", 3))
    return f"Unknown tool: {name}"
