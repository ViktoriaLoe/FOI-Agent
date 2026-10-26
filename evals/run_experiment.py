"""Run the agent over the golden dataset in LangSmith and score every run.

    python -m evals.run_experiment --split dev [--slice published] [--limit 5]
                                   [--repetitions 2] [--no-judge] [--classify-only]

Each FOI case makes two OpenAI calls (assess and draft) and the judge one more, so start small.
`--classify-only` runs just regime classification, which makes no OpenAI calls.

The experiment records the git commit and the agent's run metadata (model, prompt version,
corpus version), and a summary by category and slice is printed and saved to evals/results/.
"""

import argparse
import os
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from evals.evaluators import LABEL_EVALUATORS, regime_correct
from evals.judge import draft_judge, judge_model_name
from evals.upload_dataset import DATASET
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate

from foi_agent import nodes, rules
from foi_agent.checks import ALL_CHECKS
from foi_agent.graph import build_graph, run_metadata
from foi_agent.prompts import PROMPT_VERSION

RESULTS = Path(__file__).parent / "results"
HEADLINE = ("no_unsafe_send", "action_correct", "regime_correct", "decision_correct")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def git_version() -> str:
    """The commit the experiment ran on, marked dirty if there were uncommitted changes."""
    dirty = "-dirty" if _git("status", "--porcelain").strip() else ""
    return _git("rev-parse", "--short", "HEAD").strip() + dirty


def full_target(pause: float):
    graph = build_graph(checkpointer=InMemorySaver())

    def target(inputs: dict) -> dict:
        state = graph.invoke(inputs, {"configurable": {"thread_id": str(uuid.uuid4())}})
        time.sleep(pause)
        awaiting_review = "__interrupt__" in state
        return {k: v for k, v in state.items() if k != "__interrupt__"} | {
            "awaiting_review": awaiting_review
        }

    return target


def classify_target(inputs: dict) -> dict:
    """Regime classification and the routing decision only: no OpenAI calls."""
    state = {"request": inputs["request"]}
    state |= nodes.classify_regime(state)
    if rules.route_by_regime(state) == "route_non_foi":
        state |= rules.route_non_foi(state)
    return state


def summarise(results, title: str) -> str:
    """Mean score per evaluator overall, by category, and by slice for the headline checks."""
    overall, by_group = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in results:
        metadata = row["example"].metadata or {}
        for result in row["evaluation_results"]["results"]:
            if result.score is None:
                continue
            overall[result.key].append(result.score)
            by_group[("category", metadata.get("category"))][result.key].append(result.score)
            by_group[("slice", metadata.get("slice"))][result.key].append(result.score)

    def mean(scores):
        return f"{sum(scores) / len(scores):.0%} (n={len(scores)})" if scores else "-"

    lines = [f"# {title}", "", "| Evaluator | Score |", "| --- | --- |"]
    lines += [f"| `{key}` | {mean(scores)} |" for key, scores in overall.items()]
    for group in ("category", "slice"):
        names = sorted(name for kind, name in by_group if kind == group)
        lines += ["", f"| {group.capitalize()} | " + " | ".join(HEADLINE) + " |"]
        lines += ["| --- " * (len(HEADLINE) + 1) + "|"]
        for name in names:
            scores = by_group[(group, name)]
            lines.append(f"| {name} | " + " | ".join(mean(scores[k]) for k in HEADLINE) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--slice", help="Only examples in this slice.")
    parser.add_argument("--limit", type=int, help="Only the first N examples.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--classify-only", action="store_true", help="Classification only.")
    parser.add_argument("--pause", type=float, default=0, help="Seconds between examples.")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    load_dotenv()
    # Replies "sent" during an eval go to a throwaway folder, not the real outbox.
    os.environ.setdefault("FOI_OUTPUT_DIR", tempfile.mkdtemp(prefix="foi-eval-"))

    examples = [
        ex
        for ex in Client().list_examples(dataset_name=DATASET, splits=[args.split])
        if not args.slice or ex.metadata.get("slice") == args.slice
    ][: args.limit]

    if args.classify_only:
        target, evaluators, mode = (
            classify_target,
            [regime_correct],
            "classify",
        )
    else:
        target, mode = full_target(args.pause), "full"
        evaluators = [*ALL_CHECKS, *LABEL_EVALUATORS] + ([] if args.no_judge else [draft_judge])

    metadata = run_metadata() | {
        "git": git_version(),
        "split": args.split,
        "slice": args.slice,
        "mode": mode,
        "judge_model": None if args.no_judge or args.classify_only else judge_model_name(),
    }
    print(f"Running {len(examples)} {args.split} examples ({mode}) x {args.repetitions}")
    results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=f"{PROMPT_VERSION}-{mode}-{args.split}",
        description=args.description,
        metadata=metadata,
        num_repetitions=args.repetitions,
        max_concurrency=1,  # one request at a time, to stay within rate limits
    )

    summary = summarise(results, results.experiment_name)
    print("\n" + summary)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{results.experiment_name}.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
