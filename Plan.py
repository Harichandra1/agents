from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain_protocol.protocol import StateCommand
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage,HumanMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END, state
from dotenv import load_dotenv
load_dotenv()  # reads .env in the working directory

model = init_chat_model(
    "openai:gpt-5.4-mini"
)
planner_model = init_chat_model(
    "openai:gpt-5.4-mini"
)
class PlannerState(TypedDict):
    plan:list[str]

planner_model_with_structured_output = planner_model.with_structured_output(PlannerState)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    plan:list[str]
    normal_response : str

def planner_agent(state:MessagesState)->MessagesState:
    """LLM Plans it self"""
    plan = planner_model_with_structured_output.invoke(
        [
            SystemMessage(
                content="You are planner agent , you plan tasks"
            )
        ]
        + state["messages"]
    )
    return {"plan": plan["plan"]}
    # return state
   


def human_feedback(state:MessagesState)->Literal["planner_agent", "executor_agent"]:
    """Evaluator evaluates the tasks"""
    print("--------------------------------")
    print("Are you okay with this plan?")
    print("--------------------------------")
    for p in state["plan"]:
        print(p)
    print("--------------------------------")
    print("Enter 'yes' or 'no'")
    print("--------------------------------")
    print()
    response = input("Enter 'yes' or 'no': by looking at the plan:")
    if response == "yes":
        return "executor_agent"
    else:
        return "planner_agent"

def normal_agent(state:MessagesState)-> MessagesState :
    """This is a normal agent without plan mode"""
    response = model.invoke(
        [
            SystemMessage(
                content="You are executor agent"
            )
        ]
        +state["messages"]
    )
    print("Normal agent finished:", response.content[:200])  # actually show something
    return {"normal_response": response.content}

def executor_agent(state: MessagesState) -> MessagesState:
    """Executor executes the tasks"""
    plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are executor agent. Execute the user's request by following this plan:\n"
                    f"{plan_text}\n\n"
                    "Address each step and produce a final, complete answer."
                )
            )
        ]
        + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("planner_agent", planner_agent)
agent_builder.add_node("executor_agent", executor_agent)
agent_builder.add_edge(START, "planner_agent")
agent_builder.add_node("normal_agent",normal_agent)
agent_builder.add_edge(START,"normal_agent")

agent_builder.add_conditional_edges(
    "planner_agent",
    human_feedback,
    ["planner_agent", "executor_agent"]
)
agent_builder.add_edge("executor_agent", END)
agent_builder.add_edge("normal_agent", END)

agent = agent_builder.compile()

from IPython.display import Image, display
try:
    graph_png = agent.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as file:
        file.write(graph_png)
except Exception as e:
    print("Skipping graph image:", e)

messages = [HumanMessage(content="""
    I want to choose a laptop under ₹80,000.
    Compare a few options and recommend one.
""")]

res = agent.invoke({"messages": messages})
with open("Output1.md","w") as file:
    file.write(str(res["messages"]))
normal_messages = res["normal_response"]
with open("Output2.md","w") as file:
    file.write(normal_messages)
