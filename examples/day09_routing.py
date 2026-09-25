from langchain_core.messages import HumanMessage
from harness.routing import build_loop_graph


def main() -> None:
    question = """
    A company has 100 employees. Each employee independently has a 60% probability of being available on any given day. You need at least 70 available employees to operate normally.

    Part 1: What is the probability that the company can operate normally on a randomly chosen day?

    Part 2: The company wants this probability to be at least 99%. Without changing the 60% individual availability probability, how many employees must it employ?

    Part 3: Now suppose employees are not independent: on 20% of days, a company-wide event occurs that makes every employee 10 percentage points less likely to be available. On the remaining 80% of days, the original 60% probability applies. With 100 employees, what is the probability of having at least 70 available employees?

    Part 4: Explain why simply plugging the new average availability probability into a binomial distribution gives the wrong answer.
    """

    print("\n=== same thing as a StateGraph ===")
    config={
        "configurable":{
            "thread_id": "day09_memory",
        }
    }
    graph = build_loop_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "llm_calls": 0},
        context={"user_id":"hari-123"},
        config=config
    )
    for message in result["messages"]:
        message.pretty_print()
    print(f"\nllm_calls: {result['llm_calls']}")
    print(f"cost: ${result.get('cost', 0.0):.6f}")

if __name__ == "__main__":
    main()
