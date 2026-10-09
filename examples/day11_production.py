import base64
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from harness.production import Context, build_loop_graph

load_dotenv()

# --- OpenTelemetry setup ---
pk, sk = os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]
host = os.getenv("LANGFUSE_HOST") or os.environ["LANGFUSE_BASE_URL"]
auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()

provider = TracerProvider(resource=Resource.create({"service.name": "day11-production-agent"}))
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )
    )
)
trace.set_tracer_provider(provider)
LangChainInstrumentor().instrument(tracer_provider=provider)
tracer = trace.get_tracer("day11-production")

USER_ID = "hari-123"
THREAD_ID = "day11_production"


def main() -> None:
    question = """
    A company has 100 employees. Each employee independently has a 60% probability of being
    available on any given day. You need at least 70 available employees to operate normally.

    Part 1: What is the probability that the company can operate normally on a randomly chosen day?

    Part 2: The company wants this probability to be at least 99%. Without changing the 60%
    individual availability probability, how many employees must it employ?

    Part 3: Now suppose employees are not independent: on 20% of days, a company-wide event
    occurs that makes every employee 10 percentage points less likely to be available. On the
    remaining 80% of days, the original 60% probability applies. With 100 employees, what is the
    probability of having at least 70 available employees?

    Part 4: Explain why simply plugging the new average availability probability into a binomial
    distribution gives the wrong answer.

    Use the Python sandbox tool for the probability calculations and employee-count search.
    Include the Python code's results in your explanation and distinguish the independent and
    company-wide-event cases.
    """

    graph = build_loop_graph()

    with tracer.start_as_current_span("agent-run") as span:
        span.set_attribute("langfuse.user.id", USER_ID)
        span.set_attribute("langfuse.session.id", THREAD_ID)
        span.set_attribute("input.value", question)

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)], "llm_calls": 0},
            context=Context(user_id=USER_ID),
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        span.set_attribute("output.value", str(result["messages"][-1].content))
        span.set_attribute("agent.llm_calls", result["llm_calls"])

    for message in result["messages"]:
        message.pretty_print()
    print(f"\nllm_calls: {result['llm_calls']}")

    provider.force_flush()


if __name__ == "__main__":
    main()