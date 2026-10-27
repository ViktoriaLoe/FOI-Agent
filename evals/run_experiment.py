"""Run the agent over the golden dataset in LangSmith and score every run.

    python -m evals.run_experiment --split dev [--slice published] [--limit 5]
                                   [--repetitions 2] [--no-judge] [--classify-only]
                                   [--no-search-tool]
    python -m evals.run_experiment --rescore EXPERIMENT [--only KEY ...] [--judge]

Each FOI case makes three OpenAI calls (the regime check, assess and draft), one to three more
for the search tool, and one for the judge, so start small. `--no-search-tool` runs the agent
without the search tool, the other half of the search tool experiment. `--classify-only` runs
just regime classification, which makes no OpenAI calls.

The experiment records the agent's run metadata (model, prompt version, corpus version) and the
checks and judge versions, and LangSmith adds the git commit. A summary by category and slice is
printed and saved to evals/results/.

`--rescore` applies the current checks and label evaluators to an existing experiment's runs
without running the agent again, so it makes no OpenAI calls unless `--judge` is given. `--only`
limits it to the named evaluators, so scores that haven't changed aren't written twice.
"""

import argparse
import os
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from evals.evaluators import COUNT_KEYS, LABEL_EVALUATORS, TRAJECTORY_EVALUATORS, regime_correct
from evals.judge import JUDGE_VERSION, draft_judge, judge_model_name
from evals.upload_dataset import DATASET
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate, evaluate_existing

from foi_agent import nodes
from foi_agent.checks import ALL_CHECKS, checks_version
from foi_agent.graph import build_graph, run_metadata, search_tool_enabled
from foi_agent.prompts import PROMPT_VERSION

RESULTS = Path(__file__).parent / "results"
HEADLINE = ("no_wrong_auto_action", "action_correct", "regime_correct", "decision_correct")


def full_target(pause: float, search_tool: bool):
    graph = build_graph(checkpointer=InMemorySaver(), search_tool=search_tool)

    def target(inputs: dict) -> dict:
        state = graph.invoke(inputs, {"configurable": {"thread_id": str(uuid.uuid4())}})
        time.sleep(pause)
        awaiting_review = "__interrupt__" in state
        return {k: v for k, v in state.items() if k != "__interrupt__"} | {
            "awaiting_review": awaiting_review
        }

    return target


def classify_target(inputs: dict) -> dict:
    """The regime classifier on its own: no OpenAI calls.

    Routing isn't run, because it now also needs the independent regime check, an OpenAI call.
    """
    return nodes.classify_regime({"request": inputs["request"]})


def full_evaluators(judge: bool, only=None) -> list:
    """The checks and label evaluators, plus the judge if wanted, limited to `only` if given."""
    evaluators = [*ALL_CHECKS, *LABEL_EVALUATORS, *TRAJECTORY_EVALUATORS]
    evaluators += [draft_judge] if judge else []
    if not only:
        return evaluators
    unknown = set(only) - {e.__name__ for e in evaluators}
    if unknown:
        raise SystemExit(f"Unknown evaluators: {', '.join(sorted(unknown))}")
    return [e for e in evaluators if e.__name__ in only]


def evaluator_versions(evaluators) -> dict:
    """The checks and judge versions behind these scores, None for any not used."""
    return {
        "checks_version": checks_version() if set(evaluators) & set(ALL_CHECKS) else None,
        "judge_version": JUDGE_VERSION if draft_judge in evaluators else None,
    }


def summarise(results, title: str) -> str:
    """Mean score per evaluator overall, by category, and by slice for the headline checks.

    Pass or fail scores are shown as percentages, and counts as plain means.
    """
    overall, by_group = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in results:
        metadata = row["example"].metadata or {}
        for result in row["evaluation_results"]["results"]:
            if result.score is None:
                continue
            overall[result.key].append(result.score)
            by_group[("category", metadata.get("category"))][result.key].append(result.score)
            by_group[("slice", metadata.get("slice"))][result.key].append(result.score)

    def mean(scores, count=False):
        if not scores:
            return "-"
        average = sum(scores) / len(scores)
        return f"{average:.2f} (n={len(scores)})" if count else f"{average:.0%} (n={len(scores)})"

    lines = [f"# {title}", "", "| Evaluator | Score |", "| --- | --- |"]
    lines += [f"| `{key}` | {mean(scores, key in COUNT_KEYS)} |" for key, scores in overall.items()]
    for group in ("category", "slice"):
        names = sorted(name for kind, name in by_group if kind == group)
        lines += ["", f"| {group.capitalize()} | " + " | ".join(HEADLINE) + " |"]
        lines += ["| --- " * (len(HEADLINE) + 1) + "|"]
        for name in names:
            scores = by_group[(group, name)]
            lines.append(f"| {name} | " + " | ".join(mean(scores[k]) for k in HEADLINE) + " |")
    return "\n".join(lines) + "\n"


def save_summary(results, name: str) -> None:
    summary = summarise(results, name)
    print("\n" + summary)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(summary, encoding="utf-8")


def rescore(experiment: str, evaluators) -> None:
    """Score an existing experiment's runs with these evaluators, without running the agent."""
    print(f"Rescoring {experiment} with {', '.join(e.__name__ for e in evaluators)}")
    client = Client()
    project = client.read_project(project_name=experiment)
    results = evaluate_existing(project, evaluators=evaluators, client=client, max_concurrency=1)
    results.wait()
    # LangSmith overwrites the experiment's git and dataset metadata with the current ones. Put
    # back what it ran with, and record the rescore's versions beside it.
    metadata = project.metadata | {"rescore": evaluator_versions(evaluators)}
    client.update_project(project.id, metadata=metadata)
    # The summary covers only the evaluators just applied, so it's saved beside the original.
    save_summary(results, f"{experiment}-rescore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--slice", help="Only examples in this slice.")
    parser.add_argument("--limit", type=int, help="Only the first N examples.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--classify-only", action="store_true", help="Classification only.")
    parser.add_argument(
        "--no-search-tool", action="store_true", help="Run the agent without the search tool."
    )
    parser.add_argument("--pause", type=float, default=0, help="Seconds between examples.")
    parser.add_argument("--description", default="")
    parser.add_argument("--rescore", metavar="EXPERIMENT", help="Score an existing experiment.")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="With --rescore: only these.")
    parser.add_argument("--judge", action="store_true", help="With --rescore: add the judge.")
    args = parser.parse_args()

    load_dotenv()
    if args.rescore:
        rescore(args.rescore, full_evaluators(args.judge, args.only))
        return

    # Replies "sent" during an eval go to a throwaway folder, not the real outbox.
    os.environ.setdefault("FOI_OUTPUT_DIR", tempfile.mkdtemp(prefix="foi-eval-"))

    examples = [
        ex
        for ex in Client().list_examples(dataset_name=DATASET, splits=[args.split])
        if not args.slice or ex.metadata.get("slice") == args.slice
    ][: args.limit]

    # FOI_SEARCH_TOOL=0 turns the tool off too, so the metadata always matches the graph.
    search_tool = search_tool_enabled() and not args.no_search_tool
    if args.classify_only:
        target, evaluators, mode = classify_target, [regime_correct], "classify"
    else:
        target, mode = full_target(args.pause, search_tool), "full"
        evaluators = full_evaluators(judge=not args.no_judge)

    metadata = (
        run_metadata(search_tool)
        | {
            "split": args.split,
            "slice": args.slice,
            "mode": mode,
            "judge_model": judge_model_name() if draft_judge in evaluators else None,
        }
        | evaluator_versions(evaluators)
    )
    prefix = f"{PROMPT_VERSION}-{mode}-{args.split}" + ("" if search_tool else "-notool")
    print(f"Running {len(examples)} {args.split} examples ({mode}) x {args.repetitions}")
    results = evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=prefix,
        description=args.description,
        metadata=metadata,
        num_repetitions=args.repetitions,
        max_concurrency=1,  # one request at a time, to stay within rate limits
    )
    save_summary(results, results.experiment_name)


if __name__ == "__main__":
    main()
