"""
Day 3 — fan-out to parallel branches, then judge the results.

    START --Send(one per contender)--> agent --> judge --> select --> END

`Send` gives each branch its own isolated state, and the `transcript` reducer
(`operator.add`) merges the parallel returns back together. The judge uses
structured output rather than parsing free text -- the original free-text
version needed a hand-rolled line parser that silently dropped unparseable
scores.

Reused in Week 4 as best-of-N patch generation + judge, which is why the
contenders are a parameter rather than a module-level constant.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from harness.config import get_chat_model

DEFAULT_CONTENDERS = ["ram_charan", "ntr", "allu_arjun", "prabhas"]


class FanoutState(TypedDict):
    task: str
    contenders: list[str]
    transcript: Annotated[list[dict], operator.add]
    scores: dict[str, int]
    winner: str


class BranchState(TypedDict):
    """Isolated state handed to one parallel branch."""

    name: str
    task: str


class ContenderScore(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)


class JudgeOutput(BaseModel):
    scores: list[ContenderScore]


def format_transcript(transcript: list[dict]) -> str:
    return "\n\n".join(
        f"{i + 1}. {entry['name']}: {entry['text']}"
        for i, entry in enumerate(transcript)
    )


def build_fanout_graph(llm=None, judge_llm=None):
    llm = llm or get_chat_model()
    judge_structured = (judge_llm or llm).with_structured_output(JudgeOutput)

    def fanout(state: FanoutState) -> list[Send]:
        """One Send per contender -> the branches run in parallel."""
        contenders = state.get("contenders") or DEFAULT_CONTENDERS
        return [
            Send("agent", {"name": name, "task": state["task"]}) for name in contenders
        ]

    def agent(state: BranchState) -> dict:
        """One branch. Returns a delta the reducer merges with its siblings."""
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        f"You are {state['name']}. Passionately argue why you "
                        "are the best Telugu hero."
                    )
                ),
                HumanMessage(content=state["task"]),
            ]
        )
        return {"transcript": [{"name": state["name"], "text": response.content}]}

    def judge(state: FanoutState) -> dict:
        """Score every branch with structured output, not free-text parsing."""
        result = judge_structured.invoke(
            [
                SystemMessage(
                    content=(
                        "Score each argument out of 100 for persuasiveness and passion."
                    )
                ),
                HumanMessage(content=format_transcript(state["transcript"])),
            ]
        )
        return {"scores": {s.name: s.score for s in result.scores}}

    def select_winner(state: FanoutState) -> dict:
        """Pick the top score. Returns a delta -- never mutates state."""
        scores = state.get("scores") or {}
        if not scores:
            return {"winner": ""}
        return {"winner": max(scores, key=scores.get)}

    builder = StateGraph(FanoutState)
    builder.add_node("agent", agent)
    builder.add_node("judge", judge)
    builder.add_node("select", select_winner)

    builder.add_conditional_edges(START, fanout, ["agent"])
    builder.add_edge("agent", "judge")
    builder.add_edge("judge", "select")
    builder.add_edge("select", END)
    return builder.compile()
