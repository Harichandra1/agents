"""Offline smoke test for the Day 13 ReAct coding loop."""
from langchain_core.messages import AIMessage, HumanMessage

from harness.agent import build_loop_graph
from harness.tools.coding import build_file_tools
from harness.tools.stubs import code_exec_tool
from tests.conftest import tool_call


def test_coding_loop_routes_through_tool_node(scripted):
    llm = scripted(
        [
            {"plan": ["run the calculation"]},
            AIMessage(
                content="",
                tool_calls=[tool_call("code_exec_tool", {"code": "print(2+2)"})],
            ),
            AIMessage(content="4"),
        ]
    )
    result = build_loop_graph(llm=llm, tools=[code_exec_tool]).invoke(
        {"messages": [HumanMessage(content="What is 2+2?")], "llm_calls": 0}
    )

    assert result["plan"] == ["run the calculation"]
    assert any(
        m.type == "tool" and "executed" in m.content for m in result["messages"]
    )
    assert result["llm_calls"] == 3
    assert result["messages"][-1].content == "4"


def test_coding_loop_write_lands_in_workspace(scripted, tmp_path):
    llm = scripted(
        [
            {"plan": ["write src/hello.py"]},
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("write", {"path": "src/hello.py", "content": "x = 1\n"})
                ],
            ),
            AIMessage(content="Wrote src/hello.py"),
        ]
    )
    tools = build_file_tools(tmp_path)
    result = build_loop_graph(llm=llm, tools=tools, workspace=tmp_path).invoke(
        {"messages": [HumanMessage(content="write hello.py")], "llm_calls": 0}
    )

    assert result["plan"] == ["write src/hello.py"]
    assert (tmp_path / "src" / "hello.py").read_text() == "x = 1\n"
    assert any(m.type == "tool" and "Wrote" in m.content for m in result["messages"])


def test_llm_call_cap_stops_the_loop(scripted):
    llm = scripted(
        [
            {"plan": ["keep calling tools"]},
            AIMessage(
                content="",
                tool_calls=[tool_call("code_exec_tool", {"code": "print(1)"})],
            ),
        ]
    )
    result = build_loop_graph(llm=llm, tools=[code_exec_tool], max_llm_calls=3).invoke(
        {"messages": [HumanMessage(content="loop")], "llm_calls": 0}
    )
    assert result["llm_calls"] == 3


def test_agent_graph_close_is_idempotent(scripted):
    llm = scripted(
        [
            {"plan": ["done"]},
            AIMessage(content="ok"),
        ]
    )
    graph = build_loop_graph(llm=llm, tools=[code_exec_tool])
    graph.invoke({"messages": [HumanMessage(content="hi")], "llm_calls": 0})
    graph.close()
    graph.close()


def test_review_prompt_adds_one_self_review_turn(scripted):
    llm = scripted(
        [
            {"plan": ["answer"]},
            AIMessage(content="first answer"),
            AIMessage(content="", tool_calls=[tool_call("code_exec_tool", {"code": "1"})]),
            AIMessage(content="reviewed answer"),
        ]
    )
    result = build_loop_graph(
        llm=llm, tools=[code_exec_tool], review_prompt="Check your work."
    ).invoke({"messages": [HumanMessage(content="hi")], "llm_calls": 0})

    humans = [m.content for m in result["messages"] if m.type == "human"]
    assert humans == ["hi", "Check your work."]
    assert result["reviewed"] is True
    assert result["messages"][-1].content == "reviewed answer"


def test_no_review_prompt_ends_on_first_stop(scripted):
    llm = scripted([{"plan": ["answer"]}, AIMessage(content="done")])
    result = build_loop_graph(llm=llm, tools=[code_exec_tool]).invoke(
        {"messages": [HumanMessage(content="hi")], "llm_calls": 0}
    )
    assert result["messages"][-1].content == "done"
    assert not result.get("reviewed")
