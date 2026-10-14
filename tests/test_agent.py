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
