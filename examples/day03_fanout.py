"""Day 3 demo — parallel branches scored by a judge."""
from harness.fanout import DEFAULT_CONTENDERS, build_fanout_graph


def main() -> None:
    result = build_fanout_graph().invoke(
        {
            "task": (
                "Who is the best Telugu hero among these four: "
                "Ram Charan, NTR, Allu Arjun, and Prabhas?"
            ),
            "contenders": DEFAULT_CONTENDERS,
            "transcript": [],
            "scores": {},
            "winner": "",
        }
    )

    print("=== scores ===")
    for name, score in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {score}")
    print(f"\nwinner: {result['winner']}")


if __name__ == "__main__":
    main()
