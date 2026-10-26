"""Score traced runs with the code checks and attach the results to them in LangSmith.

    python evals/score_traces.py [--project foi-agent-local] [--limit 50] [--dry-run]

This is the online half of evaluation: it scores real runs, which have no labels, using only the
checks that don't need them. It reads existing traces, so it makes no model calls.

Re-running it replaces earlier scores instead of adding to them. Each score's id is derived from
the run and the check, and LangSmith overwrites feedback created with an existing id. Checks that
don't apply to a run (score None) are skipped. Each score records `checks_version`, so scores from
changed checks can be told apart.
"""

import argparse
import os
import uuid
from collections import defaultdict

from dotenv import load_dotenv
from langsmith import Client

from foi_agent.checks import checks_version, run_checks

# LangSmith files traces sent with LANGSMITH_AGENT_ID=foi-agent and
# LANGSMITH_AGENT_ENVIRONMENT=local under this project.
DEFAULT_PROJECT = "foi-agent-local"


def feedback_id(run_id, key: str) -> uuid.UUID:
    """The same run and check always give the same id, so re-scoring overwrites."""
    return uuid.uuid5(uuid.UUID(str(run_id)), key)


def is_scorable(run) -> bool:
    """A root run of this graph that finished with case state to check."""
    outputs = run.outputs or {}
    return run.error is None and "request" in outputs and "tracing-check" not in (run.tags or [])


def score_run(client, run, dry_run: bool = False) -> list[dict]:
    """Run the checks on one traced run and, unless dry_run, write each result as feedback."""
    results = [result for result in run_checks(run.outputs) if result["score"] is not None]
    if not dry_run:
        for result in results:
            client.create_feedback(
                run.id,
                result["key"],
                score=result["score"],
                comment=result["comment"] or None,
                feedback_id=feedback_id(run.id, result["key"]),
                feedback_source_type="model",
                source_info={"evaluator": "checks", "checks_version": checks_version()},
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=os.environ.get("LANGSMITH_PROJECT") or DEFAULT_PROJECT)
    parser.add_argument("--limit", type=int, default=50, help="Most recent root runs to read.")
    parser.add_argument("--dry-run", action="store_true", help="Print scores without writing.")
    args = parser.parse_args()

    load_dotenv()
    client = Client()
    runs = [
        run
        for run in client.list_runs(project_name=args.project, is_root=True, limit=args.limit)
        if is_scorable(run)
    ]

    tally = defaultdict(lambda: [0, 0])  # key -> [passed, applied]
    failures = []
    for run in runs:
        for result in score_run(client, run, dry_run=args.dry_run):
            tally[result["key"]][0] += result["score"]
            tally[result["key"]][1] += 1
            if result["score"] == 0:
                failures.append((run, result))

    action = "Checked" if args.dry_run else "Scored"
    print(f"{action} {len(runs)} runs in {args.project} (checks {checks_version()})\n")
    for key, (passed, applied) in tally.items():
        print(f"  {key:<28} {passed}/{applied} passed")
    for run, result in failures:
        print(f"\nFAIL {result['key']}: {result['comment']}")
        print(f"  {client.get_run_url(run=run, project_name=args.project)}")


if __name__ == "__main__":
    main()
