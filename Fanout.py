from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# model = init_chat_model("openai-gpt-5.4-mini")
model = init_chat_model("openai:gpt-5.4-mini")
def call_llm(system_prompt: str, user_content: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    res = model.invoke(messages)
    return res.content

hero_agents = [
        "ram_charan",
        "ntr",
        "allu_arjun",
        "prabhas"
    ]
class MessagesState(TypedDict):
    task: str
    Transcript: Annotated[list, operator.add]
    scores: dict
    winner: str
class InternalState(TypedDict):
    HeroName: str
    task: str   
def format_transcript(transcript: list[dict]) -> str:
    return "\n\n".join(
        f"{i+1}. {t['HeroName']}: {t['text']}" for i, t in enumerate(transcript)
    )

def fanout(state: MessagesState):
    return [Send("agent", {"HeroName": h, "task": state["task"]}) for h in hero_agents]

def agent(state: InternalState):
    response = call_llm(
        f"You are {state['HeroName']}. Passionately argue why you are the best Telugu hero.",
        state["task"],
    )
    return {"Transcript": [{"HeroName": state["HeroName"], "text": response}]}

class HeroScore(BaseModel):
    hero_name: str
    score: int = Field(ge=0, le=100)

class JudgeOutput(BaseModel):
    scores: list[HeroScore]

judge_model = model.with_structured_output(JudgeOutput)

def judge(state: MessagesState) -> MessagesState:
    transcript_text = format_transcript(state["Transcript"])
    result: JudgeOutput = judge_model.invoke([
        SystemMessage(content="Score each hero's argument out of 100 for persuasiveness and passion."),
        HumanMessage(content=transcript_text),
    ])
    state["scores"] = {s.hero_name: s.score for s in result.scores}
    return state
# def judge(state: MessagesState) -> MessagesState:
#     transcript_text = format_transcript(state["Transcript"])
#     response = call_llm(
#         "Score each numbered entry out of 100 for persuasiveness and passion. "
#         "Reply with ONLY one line per entry, formatted exactly as 'number: score'. "
#         "No extra commentary.",
#         transcript_text,
#     )
#     scores = {}
#     for line in response.strip().splitlines():
#         if ":" not in line:
#             continue
#         idx, score = line.split(":", 1)
#         idx, score = idx.strip(), "".join(ch for ch in score if ch.isdigit())
#         if idx.isdigit() and score:
#             position = int(idx) - 1
#             if 0 <= position < len(state["Transcript"]):
#                 hero_name = state["Transcript"][position]["HeroName"]
#                 scores[hero_name] = int(score)
#     state["scores"] = scores
#     return state

def select_winner(state: MessagesState) -> MessagesState:
    scores = state["scores"]
    if not scores:
        print("Judge returned no parseable scores.")
        return state
    winner = max(scores, key=scores.get)
    state["winner"] = winner
    print(f"Scores: {scores}")
    print(f"Winner: {winner}")
    return state


agent_builder = StateGraph(MessagesState)
# adding all nodes

agent_builder.add_node("agent",agent)
agent_builder.add_node("judge",judge)
agent_builder.add_node("select",select_winner)

# adding edges
agent_builder.add_conditional_edges(
    START,
    fanout,
    ["agent"],
)
agent_builder.add_edge("agent","judge")
agent_builder.add_edge("judge","select")
agent_builder.add_edge("select",END)

graph = agent_builder.compile()

try:
    graph_png = graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as file:
        file.write(graph_png)
except Exception as e:
    print("Skipping graph image:", e)

message = graph.invoke({
    "task": "Who is the best Telugu hero among these four: Ram Charan, NTR, Allu Arjun, and Prabhas?",
    "Transcript": [],
    "scores": {},
    "winner": ""
})
print(message)
