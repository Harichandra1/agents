"""
Offline smoke tests for Days 1-4.

Each test drives the day's graph with a scripted LLM and asserts it compiles,
routes, and produces the delta the next node expects. No API key, no network.

test_day04 is the important one: Day 4's tools were undecorated plain
functions and two of its three model strings were malformed, so any
delegation crashed. These tests are the regression gate for that fix.
"""
from langchain_core.messages import AIMessage, HumanMessage

from harness.fanout import build_fanout_graph
from harness.loop import build_loop_graph, run_bare_loop
from harness.planner import build_planner_graph
from harness.subagents import build_subagent_graph, get_agent_config
from harness.tools.stubs import code_exec_tool, web_search_tool
from tests.conftest import tool_call


# --------------------------------------------------------------- Day 1
def test_bare_loop_dispatches_tools_and_stops(scripted):
    llm = scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("add", {"a": 3, "b": 4})]),
            AIMessage(content="The answer is 7."),
        ]
    )
    history = run_bare_loop([HumanMessage(content="Add 3 and 4.")], llm=llm)

    tool_messages = [m for m in history if m.type == "tool"]
    assert len(tool_messages) == 1
    # The tool returns an int; content must still be a str.
    assert tool_messages[0].content == "7"
    assert isinstance(tool_messages[0].content, str)
    assert history[-1].content == "The answer is 7."


def test_loop_graph_routes_through_tool_node(scripted):
    llm = scripted(
        [
            AIMessage(content="", tool_calls=[tool_call("multiply", {"a": 6, "b": 7})]),
            AIMessage(content="42"),
        ]
    )
    result = build_loop_graph(llm=llm).invoke(
        {"messages": [HumanMessage(content="6*7?")], "llm_calls": 0}
    )

    assert any(m.type == "tool" and m.content == "42" for m in result["messages"])
    assert result["llm_calls"] == 2
    assert result["messages"][-1].content == "42"


# --------------------------------------------------------------- Day 2
def test_planner_executes_when_plan_approved(scripted):
    planner = scripted([{"plan": ["step one", "step two"]}])
    executor = scripted(
        [AIMessage(content="normal answer"), AIMessage(content="planned answer")]
    )
    graph = build_planner_graph(
        llm=executor, planner_llm=planner, approve=lambda plan: True
    )

    result = graph.invoke({"messages": [HumanMessage(content="pick a laptop")], "llm_calls": 0})

    assert result["plan"] == ["step one", "step two"]
    # Both arms ran: the planned path and the reactive control arm.
    assert result["normal_response"] == "normal answer"
    assert result["messages"][-1].content == "planned answer"


def test_planner_gate_is_injectable_without_stdin(scripted):
    """The human gate must be a parameter, not a hardcoded input() call."""
    seen = {}

    def approve(plan):
        seen["plan"] = plan
        return True

    planner = scripted([{"plan": ["only step"]}])
    executor = scripted([AIMessage(content="a"), AIMessage(content="b")])
    build_planner_graph(llm=executor, planner_llm=planner, approve=approve).invoke(
        {"messages": [HumanMessage(content="go")], "llm_calls": 0}
    )

    assert seen["plan"] == ["only step"]


# --------------------------------------------------------------- Day 3
def test_fanout_merges_branches_and_picks_winner(scripted):
    from harness.fanout import ContenderScore, JudgeOutput

    llm = scripted([AIMessage(content="I am the best.")])
    judge = scripted(
        [
            JudgeOutput(
                scores=[
                    ContenderScore(name="alpha", score=70),
                    ContenderScore(name="beta", score=91),
                ]
            )
        ]
    )
    result = build_fanout_graph(llm=llm, judge_llm=judge).invoke(
        {
            "task": "who is best?",
            "contenders": ["alpha", "beta"],
            "transcript": [],
            "scores": {},
            "winner": "",
        }
    )

    # Both parallel branches merged through the operator.add reducer.
    assert len(result["transcript"]) == 2
    assert {e["name"] for e in result["transcript"]} == {"alpha", "beta"}
    assert result["winner"] == "beta"


# --------------------------------------------------------------- Day 4
def test_day4_stubs_are_real_tools():
    """Regression: these were undecorated functions, so tool.name raised."""
    for tool in (web_search_tool, code_exec_tool):
        assert hasattr(tool, "name")
        assert hasattr(tool, "invoke")
        assert tool.description

    assert "stub" in web_search_tool.invoke({"query": "langgraph"})


def test_day4_registry_models_are_resolvable():
    """Regression: 'openai-gpt-5.4-mini' failed provider inference and raised."""
    for name in ("researcher", "coder", "generic"):
        model = get_agent_config(name)["model"]
        # None means "use the configured default" -- anything else must be
        # a provider-qualified id, never the malformed 'openai-...' form.
        assert model is None or ":" in model, f"{name} has an unresolvable model id"


def test_subagent_delegates_and_synthesizes(scripted):
    parent = scripted(
        [
            # First pass: decide to delegate one task.
            {
                "action": "delegate",
                "tasks": [
                    {
                        "id": "t1",
                        "agent": "researcher",
                        "instruction": "find trends",
                        "status": "pending",
                    }
                ],
                "response": None,
            },
            # Second pass: synthesize the worker's result.
            AIMessage(content="synthesized final answer"),
        ]
    )
    worker = scripted([AIMessage(content="worker findings")])

    result = build_subagent_graph(llm=parent, worker_llm=worker).invoke(
        {
            "messages": [HumanMessage(content="research AI trends")],
            "tasks": [],
            "results": [],
            "next_action": "respond",
        }
    )

    assert result["results"][0]["task_id"] == "t1"
    assert result["results"][0]["result"] == "worker findings"
    assert result["messages"][-1].content == "synthesized final answer"


def test_subagent_answers_directly_without_delegating(scripted):
    parent = scripted([{"action": "respond", "tasks": [], "response": "direct answer"}])

    result = build_subagent_graph(llm=parent).invoke(
        {
            "messages": [HumanMessage(content="what is context isolation?")],
            "tasks": [],
            "results": [],
            "next_action": "respond",
        }
    )

    assert result["results"] == []
    assert result["messages"][-1].content == "direct answer"
