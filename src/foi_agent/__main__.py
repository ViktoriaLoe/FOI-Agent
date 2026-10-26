"""Run one request through the graph from the command line.

python -m foi_agent "request text" [--received YYYY-MM-DD] [--review approve|reject]
python -m foi_agent --file examples/requests.jsonl --line 3 [--review approve|reject]
"""

import argparse
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from foi_agent.graph import build_graph


def _read_line(path: str, number: int) -> dict:
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not 1 <= number <= len(lines):
        raise SystemExit(f"--line must be between 1 and {len(lines)}")
    row = json.loads(lines[number - 1])
    return {key: row[key] for key in ("request", "received") if key in row}


def _print_state(state: dict) -> None:
    print(f"Regime:     {state.get('regime')}")
    scores = state.get("regime_scores") or {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(f"Scores:     {', '.join(f'{label} {p:.2f}' for label, p in ranked)}")
    print(f"Deadline:   {state.get('deadline')}")
    print(f"Assessment: {json.dumps(state.get('assessment'), indent=2)}")
    print(f"Action:     {state.get('action')}")
    print(f"Escalation reasons: {state.get('escalation_reasons')}")
    print(f"\nDraft:\n{state.get('draft')}\n")
    print(f"Outcome:    {state.get('outcome')}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="foi_agent", description="Triage one FOI request.")
    parser.add_argument("request", nargs="?", help="The request text.")
    parser.add_argument("--received", help="Date received, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--file", help='A JSONL file of {"request", "received"} lines.')
    parser.add_argument("--line", type=int, default=1, help="Line of --file to run (from 1).")
    parser.add_argument(
        "--review", choices=["approve", "reject"], help="Officer decision if the case escalates."
    )
    args = parser.parse_args()

    if args.file:
        inputs = _read_line(args.file, args.line)
    elif args.request:
        inputs = {"request": args.request}
    else:
        parser.error("give the request text or --file")
    if args.received:
        inputs["received"] = args.received

    load_dotenv()
    graph = build_graph(checkpointer=InMemorySaver())
    # Tag where the run came from; for sample files, record which line, so traces link back.
    metadata = {"example": f"{args.file}:{args.line}"} if args.file else {}
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["cli"],
        "metadata": metadata,
    }

    state = graph.invoke(inputs, config)
    print(f"Case id (LangSmith thread): {config['configurable']['thread_id']}")
    if "__interrupt__" in state:
        if args.review:
            state = graph.invoke(Command(resume=args.review), config)
        else:
            _print_state(state)
            print("\nAwaiting officer review. Rerun with --review approve or --review reject.")
            return
    _print_state(state)


if __name__ == "__main__":
    main()
