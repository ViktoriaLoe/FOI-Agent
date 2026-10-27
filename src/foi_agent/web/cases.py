"""Cases for the demo UI. This is the only web module that talks to LangGraph and LangSmith.

A case is one request run through the graph on its own thread (case id = thread id). Checkpoints
and a small `web_cases` table live in `web.sqlite` under FOI_OUTPUT_DIR. Everything the templates
show is a plain dict built here, so the templates hold no logic.
"""

import json
import logging
import os
import queue
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langsmith import Client

from foi_agent import rules
from foi_agent.checks import run_checks, section_number
from foi_agent.deadline import is_working_day
from foi_agent.graph import build_graph

logger = logging.getLogger(__name__)

EXAMPLES_FILE = "examples/requests.jsonl"
GOLDEN_FILE = "evals/datasets/golden.jsonl"

# (node, label, kind) in graph order.
PIPELINE = (
    ("classify_regime", "Classify regime", "judgement"),
    ("check_regime", "Check regime", "judgement"),
    ("compute_deadline", "Deadline", "rule"),
    ("route_non_foi", "Route non-FOI", "rule"),
    ("retrieve_evidence", "Evidence", "retrieval"),
    ("assess", "Assess", "judgement"),
    ("decide_action", "Send or escalate", "rule"),
    ("draft_reply", "Draft reply", "judgement"),
    ("human_review", "Officer review", "interrupt"),
    ("log_and_send", "Log and send", "rule"),
)

REGIME_LABELS = {
    "FOI": "Freedom of Information",
    "EIR": "Environmental Information Regulations",
    "SAR": "Subject access request",
    "BAU": "Business as usual",
}
STATUS_LABELS = {
    "running": "Running",
    "sent": "Sent",
    "routed": "Routed",
    "awaiting_review": "Awaiting review",
    "rejected": "Rejected",
    "error": "Error",
}
DECISION_LABELS = {
    "provide": "Provide",
    "signpost": "Signpost to published information",
    "part_refuse": "Part refuse",
    "refuse": "Refuse",
}
_WORKING_DAYS_NOTE = (
    "Twenty working days from the next working day, skipping weekends and UK bank holidays."
)
DEADLINE_NOTES = {
    "FOI": _WORKING_DAYS_NOTE,
    "EIR": _WORKING_DAYS_NOTE,
    "SAR": (
        "One calendar month: the same date next month, or its last day, moved to the next "
        "working day if it falls on a weekend or UK bank holiday."
    ),
    "BAU": "Business as usual requests have no statutory deadline.",
}
REVIEW_ACTIONS = {"approve": "Approved", "edit": "Edited and sent", "reject": "Rejected"}
SOURCE_LABELS = {
    "disclosure_log": "Disclosure log",
    "publication_scheme": "Publication scheme",
    "exemption_guidance": "Exemption guidance",
    "regime_guidance": "Regime guidance",
}
# Human label and the step it scores, as in the README's Checks table.
CHECK_INFO = {
    "evidence_found": ("Evidence found", "Retrieve evidence"),
    "citations_grounded": ("Citations grounded", "Assess"),
    "assessment_consistent": ("Assessment consistent", "Assess"),
    "exemption_sections_known": ("Exemption sections known", "Assess"),
    "draft_links_grounded": ("Draft links grounded", "Draft reply"),
    "draft_states_rights": ("Draft states rights", "Draft reply"),
    "draft_no_new_personal_data": ("No new personal data", "Draft reply"),
}


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def repo_file(relative: str) -> Path:
    """A file in the repo checkout this package was installed from (editable install), or else in
    the working directory."""
    beside_package = Path(__file__).resolve().parents[3] / relative
    return beside_package if beside_package.exists() else Path.cwd() / relative


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@cache
def golden_by_request() -> dict[str, dict]:
    """The golden dataset keyed by request text, so a run of a labelled request can be set
    against its label."""
    rows = _read_jsonl(repo_file(GOLDEN_FILE))
    return {row["inputs"]["request"].strip(): row for row in rows}


# --- Pure view-model helpers ------------------------------------------------------------------


def display_date(value: str | date | None, year: bool = True) -> str | None:
    """Like "Mon 5 Oct 2026", or "Mon 5 Oct" without the year. Built by hand because %-d (no
    leading zero) does not work on Windows."""
    if not value:
        return None
    day = date.fromisoformat(value) if isinstance(value, str) else value
    text = f"{day:%a} {day.day} {day:%b}"
    return f"{text} {day.year}" if year else text


def display_timestamp(value: str | None) -> str | None:
    """An ISO timestamp in local time, as "Mon 26 Oct 2026, 14:05"."""
    if not value:
        return None
    moment = datetime.fromisoformat(value).astimezone()
    return f"{display_date(moment.date())}, {moment:%H:%M}"


def working_days_left(deadline: date, today: date) -> int:
    """Working days after today up to and including the deadline. Negative once it has passed:
    -1 means the deadline was the last working day before today."""
    start, end = sorted((today, deadline))
    days = sum(is_working_day(start + timedelta(days=n)) for n in range(1, (end - start).days + 1))
    return days if deadline >= today else -days


def example_title(note: str, limit: int = 48) -> str:
    """The note's lead phrase: the text before its first colon, comma or full stop."""
    lead = re.split(r"[:,.]", note, maxsplit=1)[0].strip()
    if len(lead) <= limit:
        return lead
    return lead[:limit].rsplit(" ", 1)[0] + "…"


def _sections(exemptions: list) -> list[str]:
    """Exemption sections as "s40", whether given as strings or as {"section", "reason"}."""
    sections = set()
    for exemption in exemptions or []:
        section = exemption.get("section", "") if isinstance(exemption, dict) else exemption
        sections.add(f"s{section_number(section)}" if section_number(section) else section)
    return sorted(sections)


def gold_view(gold: dict, values: dict) -> dict:
    """The golden label next to what this run did, field by field.

    A field neither side has (no decision on a routed SAR, say) is left out.
    """
    labels, metadata = gold["outputs"], gold.get("metadata") or {}
    assessment = values.get("assessment") or {}
    pairs = [
        ("Regime", labels.get("regime"), values.get("regime")),
        ("Action", labels.get("action"), values.get("action")),
        ("Decision", labels.get("decision"), assessment.get("recommended_decision")),
    ]
    if labels.get("decision") or assessment:
        expected = _sections(labels.get("exemptions"))
        actual = _sections(assessment.get("exemptions"))
        pairs.append(("Exemptions", ", ".join(expected), ", ".join(actual)))
    fields = [
        {"label": label, "expected": expected, "actual": actual, "match": expected == actual}
        for label, expected, actual in pairs
        if expected or actual
    ]
    return {
        "slice": metadata.get("slice"),
        "category": metadata.get("category"),
        "why": metadata.get("why"),
        "fields": fields,
        "matched": all(field["match"] for field in fields),
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def visited_nodes(values: dict) -> set[str]:
    """The nodes that have run, worked out from the state they leave behind.

    Deriving this from state rather than from the stream means a reopened or replayed case shows
    the same pipeline as the live run did.
    """
    visited = set()
    if "regime" in values:
        visited.add("classify_regime")
    if "regime_check" in values:
        visited.add("check_regime")
    # compute_deadline sets no deadline for business as usual, but every later node follows it.
    if "deadline" in values or "action" in values or "evidence" in values:
        visited.add("compute_deadline")
    # route_non_foi sets the action before any assessment; decide_action sets it after one.
    if "action" in values and "assessment" not in values:
        visited.add("route_non_foi")
    for key, node in (
        ("evidence", "retrieve_evidence"),
        ("assessment", "assess"),
    ):
        if key in values:
            visited.add(node)
    if "assessment" in values and "action" in values:
        visited.add("decide_action")
    if "assessment" in values and "draft" in values:
        visited.add("draft_reply")
    if "review" in values:
        visited.add("human_review")
    if values.get("outcome") in ("sent", "routed"):
        visited.add("log_and_send")
    return visited


def _action_summary(values: dict) -> str:
    action = values.get("action")
    if action == "escalate":
        return "escalate · " + _plural(len(values.get("escalation_reasons") or []), "reason")
    return {"route": "routed", "send": "send"}.get(action, "")


def _node_summary(node: str, status: str, values: dict) -> str:
    if status == "paused":
        return "awaiting officer"
    if status != "done":
        return ""
    if node == "classify_regime":
        regime = values.get("regime")
        top = (values.get("regime_scores") or {}).get(regime)
        return f"{regime} {top:.2f}" if top is not None else str(regime)
    if node == "check_regime":
        check = values.get("regime_check")
        return f"{check} · {'agrees' if check == values.get('regime') else 'disagrees'}"
    if node in ("route_non_foi", "decide_action"):
        return _action_summary(values)
    if node == "compute_deadline":
        if not values.get("deadline"):
            return "no deadline"
        return f"due {display_date(values['deadline'], year=False)}"
    if node == "retrieve_evidence":
        return _plural(len(values.get("evidence") or []), "record")
    if node == "assess":
        return (values.get("assessment") or {}).get("recommended_decision") or ""
    if node == "draft_reply":
        return _plural(len((values.get("draft") or "").split()), "word")
    if node == "human_review":
        return REVIEW_ACTIONS.get((values.get("review") or {}).get("action"), "").lower()
    if node == "log_and_send":
        return values.get("outcome") or ""
    return ""


def pipeline_view(values: dict, next_nodes: tuple, finished: bool) -> list[dict]:
    """Each node's status and summary.

    `next_nodes` is `get_state().next`. While the run is going those nodes are running; once it
    has finished, unvisited nodes were skipped and a waiting human_review is paused.
    """
    visited = visited_nodes(values)
    steps = []
    for node, label, kind in PIPELINE:
        if node in visited:
            status = "done"
        elif node in next_nodes:
            status = "paused" if finished and node == "human_review" else "running"
        else:
            status = "skipped" if finished else "pending"
        summary = _node_summary(node, status, values)
        steps.append(
            {"node": node, "label": label, "kind": kind, "status": status, "summary": summary}
        )
    return steps


def next_after_finish(values: dict) -> tuple:
    """`get_state().next` for a finished run, from its state alone (for saved replays)."""
    if values.get("action") == "escalate" and "review" not in values:
        return ("human_review",)
    return ()


def case_status(values: dict, next_nodes: tuple, error: str | None = None) -> str:
    if error:
        return "error"
    if values.get("outcome") in ("sent", "routed", "rejected"):
        return values["outcome"]
    if "human_review" in next_nodes:
        return "awaiting_review"
    return "running"


def _field_value(value) -> str:
    if isinstance(value, list):
        return "; ".join(map(str, value)) or "none"
    return str(value)


def evidence_view(doc: dict, cited_ids: set[str]) -> dict:
    hidden = {"id", "title", "source", "score", "url"}
    return {
        "id": doc["id"],
        "title": doc.get("title", ""),
        "source": doc.get("source"),
        "source_label": SOURCE_LABELS.get(doc.get("source"), doc.get("source")),
        "score": round(doc.get("score", 0.0), 2),
        "url": doc.get("url"),
        "cited": doc["id"] in cited_ids,
        "fields": [[key, _field_value(value)] for key, value in doc.items() if key not in hidden],
    }


def _checks_view(values: dict) -> list[dict]:
    checks = []
    for result in run_checks(values):
        label, step = CHECK_INFO.get(result["key"], (result["key"], ""))
        # Passing checks leave an empty list behind a prefix, like "Not retrieved: ".
        comment = result["comment"].strip()
        comment = "" if comment.endswith(":") else comment
        checks.append(
            {
                "key": result["key"],
                "label": label,
                "step": step,
                "score": result["score"],
                "comment": comment,
            }
        )
    return checks


def case_view(
    case_id: str,
    values: dict,
    next_nodes: tuple,
    *,
    request: str | None = None,
    received: str | None = None,
    example_line: int | None = None,
    error: str | None = None,
    replayed_at: str | None = None,
    today: date | None = None,
) -> dict:
    """Everything result.html and review_case.html show about one case."""
    today = today or date.today()
    gold = golden_by_request().get((values.get("request") or request or "").strip())
    status = case_status(values, next_nodes, error)
    regime = values.get("regime")
    scores = values.get("regime_scores") or {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    deadline = values.get("deadline")

    evidence_docs = values.get("evidence") or []
    raw_assessment = values.get("assessment")
    cited_ids = set()
    assessment = None
    if raw_assessment:
        cited_ids = set(raw_assessment.get("cited_ids") or [])
        cited_ids |= {raw_assessment.get("published_id")} - {None}
        evidence = [evidence_view(doc, cited_ids) for doc in evidence_docs]
        decision = raw_assessment.get("recommended_decision")
        assessment = {
            "decision": decision,
            "decision_label": DECISION_LABELS.get(decision, decision),
            "already_published": raw_assessment.get("already_published"),
            "published_id": raw_assessment.get("published_id"),
            "reasoning": raw_assessment.get("reasoning"),
            "exemptions": [
                {"section": e.get("section"), "reason": e.get("reason")}
                for e in raw_assessment.get("exemptions") or []
            ],
            "cited": [doc for doc in evidence if doc["cited"]],
        }
    else:
        evidence = [evidence_view(doc, cited_ids) for doc in evidence_docs]

    review_action = (values.get("review") or {}).get("action")
    return {
        "id": case_id,
        "short_id": case_id[:8],
        "request": values.get("request") or request,
        "received_display": display_date(values.get("received") or received),
        "replayed": replayed_at is not None,
        "replayed_at_display": display_timestamp(replayed_at),
        "example_line": example_line,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "regime": regime,
        "regime_label": REGIME_LABELS.get(regime),
        "regime_scores": [
            {"label": label, "prob": prob, "pct": round(prob * 100), "is_top": i == 0}
            for i, (label, prob) in enumerate(ranked)
        ],
        "threshold_pct": round(rules.REGIME_MIN_PROB * 100),
        "regime_unclear": rules._regime_unclear(values),
        "deadline_display": display_date(deadline),
        "deadline_note": DEADLINE_NOTES.get(regime, "Worked out once the regime is classified."),
        "working_days_left": (
            working_days_left(date.fromisoformat(deadline), today) if deadline else None
        ),
        "assessment": assessment,
        "evidence": evidence,
        "action": values.get("action"),
        "escalation_reasons": values.get("escalation_reasons") or [],
        "draft": values.get("draft"),
        "review": (
            {"action": review_action, "action_label": REVIEW_ACTIONS.get(review_action)}
            if review_action
            else None
        ),
        "checks": _checks_view(values),
        "gold": gold_view(gold, values) if gold else None,
        "trace_url": f"/cases/{case_id}/trace" if tracing_enabled() else None,
        "review_url": f"/review?case={case_id}" if status == "awaiting_review" else None,
        "error": error,
    }


def record_review_feedback(run_id: str, action: str) -> None:
    """Attach the officer's decision to the triage run in LangSmith, as a label for evaluation.

    Best effort: a LangSmith failure is logged, never raised, so a review always goes through.
    """
    if not tracing_enabled():
        return
    try:
        client = Client()
        client.create_feedback(
            run_id, key="officer_review", score=int(action != "reject"), value=action
        )
        client.create_feedback(run_id, key="draft_edited", score=int(action == "edit"))
    except Exception:
        logger.warning("Could not record LangSmith feedback on run %s", run_id, exc_info=True)


# --- Storage and the graph --------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_cases (
    case_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    request TEXT NOT NULL,
    received TEXT,
    example_line INTEGER,
    created_at TEXT NOT NULL,
    started INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    decided_at TEXT
)
"""


class Cases:
    """Creates, runs, lists and resumes cases.

    The data directory and the graph are resolved on first use, so building the app needs no
    API keys and touches no files. Without a `graph`, the triage graph is built with a SqliteSaver
    over web.sqlite. Keep `data_dir` the same as FOI_OUTPUT_DIR, where log_and_send writes the case
    log this reads.
    """

    def __init__(self, graph=None, data_dir: str | Path | None = None):
        self._graph = graph
        self._data_dir = Path(data_dir) if data_dir else None
        self._lock = threading.Lock()

    @property
    def data_dir(self) -> Path:
        return self._data_dir or Path(os.environ.get("FOI_OUTPUT_DIR", "var"))

    @property
    def graph(self):
        with self._lock:
            if self._graph is None:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                # Shared across the threads FastAPI runs sync routes in; SqliteSaver locks it.
                conn = sqlite3.connect(self.data_dir / "web.sqlite", check_same_thread=False)
                self._graph = build_graph(checkpointer=SqliteSaver(conn))
            return self._graph

    def _db(self) -> sqlite3.Connection:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.data_dir / "web.sqlite")
        db.row_factory = sqlite3.Row
        db.execute(_SCHEMA)
        return db

    def _row(self, case_id: str) -> sqlite3.Row | None:
        with closing(self._db()) as db:
            return db.execute("SELECT * FROM web_cases WHERE case_id = ?", (case_id,)).fetchone()

    def _config(self, row: sqlite3.Row) -> dict:
        config = {"configurable": {"thread_id": row["case_id"]}, "tags": ["web"]}
        if row["example_line"]:
            # The same form as the CLI's, so traces link back to the sample line.
            config["metadata"] = {"example": f"{EXAMPLES_FILE}:{row['example_line']}"}
        return config

    def _state(self, row: sqlite3.Row):
        return self.graph.get_state(self._config(row))

    # Examples

    def examples(self) -> list[dict]:
        rows = _read_jsonl(repo_file(EXAMPLES_FILE))
        return [self._example_view(line, row) for line, row in enumerate(rows, start=1)]

    def example(self, line: int) -> dict | None:
        examples = self.examples()
        return examples[line - 1] if 1 <= line <= len(examples) else None

    def _demo_run_path(self, line: int) -> Path:
        return self.data_dir / "demo_runs" / f"{line}.json"

    def _example_view(self, line: int, row: dict) -> dict:
        note = row.get("note")
        demo_run = self._demo_run_path(line)
        ran_at = None
        if demo_run.exists():
            ran_at = json.loads(demo_run.read_text(encoding="utf-8"))["ran_at"]
        gold = golden_by_request().get(row["request"].strip())
        return {
            "line": line,
            "request": row["request"],
            "received": row.get("received"),
            "received_display": display_date(row.get("received")),
            "title": example_title(note) if note else f"Example {line}",
            "note": note,
            # The golden dataset's expected action, when the example is one of its requests.
            "expected_path": gold["outputs"].get("action") if gold else None,
            "has_replay": ran_at is not None,
            "replay_at_display": display_timestamp(ran_at),
        }

    # Running a case

    def create(self, request: str, received: str | None = None) -> str:
        """Record a new case. It runs when its stream is first opened."""
        request = request.strip()
        received = received or date.today().isoformat()
        example_line = next(
            (ex["line"] for ex in self.examples() if ex["request"].strip() == request), None
        )
        case_id = str(uuid.uuid4())
        with closing(self._db()) as db, db:
            db.execute(
                "INSERT INTO web_cases (case_id, run_id, request, received, example_line,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    str(uuid.uuid4()),
                    request,
                    received,
                    example_line,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
        return case_id

    def exists(self, case_id: str) -> bool:
        return self._row(case_id) is not None

    def new_run(self, case_id: str) -> dict:
        """The case and pipeline for run.html before the stream starts."""
        row = self._row(case_id)
        return {
            "case": self._case_from_row(row, {}, ("classify_regime",)),
            "pipeline": pipeline_view({}, ("classify_regime",), finished=False),
        }

    def _claim(self, case_id: str) -> bool:
        """Mark the case started. False if it already was, so a case only ever runs once."""
        with closing(self._db()) as db, db:
            cursor = db.execute(
                "UPDATE web_cases SET started = 1 WHERE case_id = ? AND started = 0", (case_id,)
            )
            return cursor.rowcount == 1

    def run(self, case_id: str) -> Iterator[tuple[str, dict]]:
        """Run the case, yielding ("pipeline", {pipeline}) after each node and finally
        ("done", {case}).

        If the case has already started (the browser reconnected, or a second tab), it isn't run
        again: this yields the pipeline and case as they stand.
        """
        row = self._row(case_id)
        if not self._claim(case_id):
            yield from self._final_events(case_id)
            return

        # The graph runs on its own thread, so a case still finishes (and reaches the review queue)
        # if the browser goes away mid-run. The stream just relays its progress.
        updates: queue.Queue = queue.Queue()
        threading.Thread(target=self._run_graph, args=(row, updates), daemon=True).start()
        yield "pipeline", {"pipeline": pipeline_view({}, ("classify_regime",), finished=False)}
        while (pipeline := updates.get()) is not None:
            yield "pipeline", {"pipeline": pipeline}
        yield from self._final_events(case_id)

    def _run_graph(self, row: sqlite3.Row, updates: queue.Queue) -> None:
        """Run one case to its end or its pause, putting the pipeline after each node on
        `updates`, then None."""
        config = self._config(row)
        inputs = {"request": row["request"], "received": row["received"]}
        # A known root run id, so the trace can be found and the officer's decision attached.
        first_config = {**config, "run_id": uuid.UUID(row["run_id"])}
        try:
            for update in self.graph.stream(inputs, first_config, stream_mode="updates"):
                if "__interrupt__" in update:
                    continue  # the pause is shown once the stream ends
                state = self.graph.get_state(config)
                updates.put(pipeline_view(state.values, state.next, finished=False))
        except Exception as exc:
            logger.exception("Case %s failed", row["case_id"])
            with closing(self._db()) as db, db:
                db.execute(
                    "UPDATE web_cases SET error = ? WHERE case_id = ?",
                    (f"{type(exc).__name__}: {exc}", row["case_id"]),
                )
        else:
            if row["example_line"]:
                self._save_demo_run(row)
        finally:
            updates.put(None)

    def _final_events(self, case_id: str) -> Iterator[tuple[str, dict]]:
        row = self._row(case_id)
        state = self._state(row)
        # A failed run stops where it failed: nothing is still running.
        next_nodes = () if row["error"] else state.next
        yield "pipeline", {"pipeline": pipeline_view(state.values, next_nodes, finished=True)}
        yield "done", {"case": self._case_from_row(row, state.values, next_nodes)}

    def _case_from_row(self, row: sqlite3.Row, values: dict, next_nodes: tuple) -> dict:
        return case_view(
            row["case_id"],
            values,
            next_nodes,
            request=row["request"],
            received=row["received"],
            example_line=row["example_line"],
            error=row["error"],
        )

    def _save_demo_run(self, row: sqlite3.Row) -> None:
        """Keep the last live run of an example, for "Show last run" without calling a model."""
        path = self._demo_run_path(row["example_line"])
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "case_id": row["case_id"],
            "run_id": row["run_id"],
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "state": self._state(row).values,
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def replay(self, line: int) -> dict | None:
        """The case and pipeline of an example's last live run, or None if it has none."""
        path = self._demo_run_path(line)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        values = record["state"]
        next_nodes = next_after_finish(values)
        return {
            "case": case_view(
                record["case_id"],
                values,
                next_nodes,
                example_line=line,
                replayed_at=record["ran_at"],
            ),
            "pipeline": pipeline_view(values, next_nodes, finished=True),
        }

    # Review

    def case(self, case_id: str) -> dict | None:
        row = self._row(case_id)
        if row is None:
            return None
        state = self._state(row)
        return self._case_from_row(row, state.values, () if row["error"] else state.next)

    def pending(self) -> list[dict]:
        """Cases paused for officer review, soonest deadline first."""
        with closing(self._db()) as db:
            rows = db.execute("SELECT * FROM web_cases WHERE started = 1").fetchall()
        items = []
        for row in rows:
            state = self._state(row)
            if "human_review" not in state.next:
                continue
            values = state.values
            reasons = values.get("escalation_reasons") or []
            deadline = values.get("deadline")
            request = values.get("request") or row["request"]
            item = {
                "id": row["case_id"],
                "short_id": row["case_id"][:8],
                "request_excerpt": request if len(request) <= 90 else request[:89] + "…",
                "regime": values.get("regime"),
                "deadline_display": display_date(deadline),
                "working_days_left": (
                    working_days_left(date.fromisoformat(deadline), date.today())
                    if deadline
                    else None
                ),
                "reasons_count": len(reasons),
                "first_reason": reasons[0] if reasons else None,
            }
            items.append((deadline, item))
        # ISO dates sort as text; cases with no deadline (routed before FOI work) go last.
        items.sort(key=lambda pair: (pair[0] is None, pair[0] or ""))
        return [item for _, item in items]

    def decided(self, limit: int = 8) -> list[dict]:
        """The officer's latest decisions, newest first.

        Read from web_cases rather than the case log, because a rejected case is never logged.
        """
        with closing(self._db()) as db:
            rows = db.execute(
                "SELECT * FROM web_cases WHERE decided_at IS NOT NULL"
                " ORDER BY decided_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            values = self._state(row).values
            items.append(
                {
                    "case_id": row["case_id"],
                    "short_id": row["case_id"][:8],
                    "outcome": values.get("outcome"),
                    "review_action": (values.get("review") or {}).get("action"),
                    "regime": values.get("regime"),
                    "timestamp_display": display_timestamp(row["decided_at"]),
                }
            )
        return items

    def review(self, case_id: str, action: str, draft: str | None = None) -> dict | None:
        """Resume a paused case with the officer's decision. None if there is no such case.

        Sending sends the reply as the officer left it: an approval whose text differs from the
        agent's draft is an edit, and an edit that changed nothing is an approval, so the
        draft_edited feedback means what it says. A case that isn't waiting (decided already,
        perhaps in another tab) is left as it is.
        """
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"Unknown review action: {action!r}")
        row = self._row(case_id)
        if row is None:
            return None
        state = self._state(row)
        if "human_review" in state.next:
            # Browsers send textarea line breaks as CRLF.
            draft = (draft or "").replace("\r\n", "\n").strip()
            if action != "reject":
                agent_draft = (state.values.get("draft") or "").strip()
                if not (draft or agent_draft):
                    # Cases escalated by route_non_foi (EIR, or blocked from acting on their
                    # own) come with no draft.
                    raise ValueError("There is no reply to send. Write one, or reject the case.")
                edited = bool(draft) and draft != agent_draft
                action = "edit" if edited else "approve"
            resume = {"action": action, "draft": draft if action == "edit" else None}
            self.graph.invoke(Command(resume=resume), self._config(row))
            with closing(self._db()) as db, db:
                db.execute(
                    "UPDATE web_cases SET decided_at = ? WHERE case_id = ?",
                    (datetime.now(UTC).isoformat(timespec="seconds"), case_id),
                )
            record_review_feedback(row["run_id"], action)
        return self.case(case_id)

    def trace_url(self, case_id: str) -> str | None:
        """The case's LangSmith trace, or None if tracing is off or the lookup fails.

        Looked up only when asked for, so pages never wait on LangSmith.
        """
        row = self._row(case_id)
        if row is None or not tracing_enabled():
            return None
        try:
            client = Client()
            run = client.read_run(row["run_id"])
            # The run carries its own project, so no project name is needed.
            return client.get_run_url(run=run)
        except Exception:
            logger.warning("Could not find the trace for case %s", case_id, exc_info=True)
            return None
