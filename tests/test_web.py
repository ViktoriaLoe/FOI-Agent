"""The demo UI: its view-model helpers, and its routes over the graph with fake LLM nodes."""

import json
import re
import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from foi_agent import nodes
from foi_agent.web import cases as cases_module
from foi_agent.web.app import create_app, sse
from foi_agent.web.cases import (
    example_title,
    gold_view,
    next_after_finish,
    pipeline_view,
    working_days_left,
)

SALARIES_REQUEST = "Please send me the senior officer salaries for 2025/26."
SAR_REQUEST = "Send me everything you hold about me."


def assessment(**overrides):
    base = {
        "reasoning": "Fake assessment.",
        "already_published": False,
        "published_id": None,
        "exemptions": [],
        "recommended_decision": "provide",
        "cited_ids": [],
    }
    return {**base, **overrides}


SIGNPOST = assessment(
    already_published=True,
    published_id="PS-003",
    recommended_decision="signpost",
    cited_ids=["PS-003"],
)
EXEMPTION = assessment(
    recommended_decision="part_refuse",
    exemptions=[{"section": "s40", "reason": "Junior staff salaries"}],
)


# --- View-model helpers -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("note", "title"),
    [
        ("Business as usual: a missed bin and a pothole report", "Business as usual"),
        ("Already published, near-identical wording to DL-2026-014", "Already published"),
        (
            "EIR worded as FOI (tree felling). Similar to DL-2025-016",
            "EIR worded as FOI (tree felling)",
        ),
        (
            "Prompt injection plus an attempt to extract personal data (s40). Must escalate",
            "Prompt injection plus an attempt to extract…",
        ),
    ],
)
def test_example_title_is_the_notes_lead_phrase(note, title):
    assert example_title(note) == title


def gold(**outputs):
    return {"outputs": outputs, "metadata": {"slice": "published", "category": "normal"}}


def test_gold_view_matches_a_run_that_did_what_the_label_says():
    label = gold(regime="FOI", action="send", decision="signpost", exemptions=["s21"])
    run = {
        "regime": "FOI",
        "action": "send",
        "assessment": {**SIGNPOST, "exemptions": [{"section": "Section 21", "reason": "..."}]},
    }

    view = gold_view(label, run)

    assert view["matched"]
    assert [f["label"] for f in view["fields"]] == ["Regime", "Action", "Decision", "Exemptions"]


def test_gold_view_marks_each_field_that_differs():
    label = gold(regime="FOI", action="escalate", decision="part_refuse", exemptions=["s40"])
    run = {"regime": "FOI", "action": "send", "assessment": SIGNPOST}

    view = gold_view(label, run)

    assert not view["matched"]
    misses = {f["label"]: (f["expected"], f["actual"]) for f in view["fields"] if not f["match"]}
    assert misses == {
        "Action": ("escalate", "send"),
        "Decision": ("part_refuse", "signpost"),
        "Exemptions": ("s40", ""),
    }


def test_gold_view_leaves_out_fields_a_routed_case_has_no_label_for():
    view = gold_view(gold(regime="SAR", action="route"), {"regime": "SAR", "action": "route"})

    assert [f["label"] for f in view["fields"]] == ["Regime", "Action"]
    assert view["matched"]


def test_examples_take_their_expected_path_from_the_golden_dataset(web):
    examples = web("SAR").app.state.cases.examples()

    # Line 4 is an EIR request whose note says "not already published": it must not read as send.
    assert {ex["line"]: ex["expected_path"] for ex in examples}[4] == "escalate"
    assert {ex["expected_path"] for ex in examples} <= {"send", "route", "escalate", "clarify"}


@pytest.mark.parametrize(
    ("deadline", "left"),
    [
        (date(2026, 10, 26), 0),  # due today
        (date(2026, 10, 30), 4),  # Tue to Fri
        (date(2026, 11, 2), 5),  # the weekend doesn't count
        (date(2026, 10, 23), -1),  # due last Friday
    ],
)
def test_working_days_left_counts_only_working_days(deadline, left):
    assert working_days_left(deadline, today=date(2026, 10, 26)) == left  # a Monday


def test_working_days_left_skips_bank_holidays():
    # Christmas Day and Boxing Day 2026 (moved to Mon 28 Dec) fall between these dates.
    assert working_days_left(date(2026, 12, 29), today=date(2026, 12, 24)) == 1


def statuses(pipeline):
    return {step["node"]: step["status"] for step in pipeline}


def test_pipeline_marks_the_next_node_running_and_the_rest_pending():
    values = {"request": "x", "regime": "FOI", "regime_scores": {"FOI": 0.9}}
    steps = statuses(pipeline_view(values, ("check_regime",), finished=False))

    assert steps["classify_regime"] == "done"
    assert steps["check_regime"] == "running"
    assert steps["compute_deadline"] == steps["route_non_foi"] == steps["assess"] == "pending"


def test_finished_pipeline_skips_unvisited_nodes_and_pauses_review():
    values = {
        "request": "x",
        "regime": "EIR",
        "regime_scores": {"EIR": 0.8},
        "regime_check": "EIR",
        "deadline": "2026-11-23",
        "action": "escalate",
        "escalation_reasons": ["Environmental information: handle under the EIR."],
    }
    pipeline = pipeline_view(values, next_after_finish(values), finished=True)
    steps = statuses(pipeline)

    assert steps["check_regime"] == steps["compute_deadline"] == steps["route_non_foi"] == "done"
    assert steps["retrieve_evidence"] == steps["assess"] == steps["log_and_send"] == "skipped"
    assert steps["human_review"] == "paused"
    summaries = {step["node"]: step["summary"] for step in pipeline}
    assert summaries["classify_regime"] == "EIR 0.80"
    assert summaries["check_regime"] == "EIR · agrees"
    assert summaries["compute_deadline"] == "due Mon 23 Nov"
    assert summaries["route_non_foi"] == "escalate · 1 reason"
    assert summaries["human_review"] == "awaiting officer"


def test_pipeline_shows_a_disagreeing_check_and_a_case_with_no_deadline():
    values = {
        "request": "x",
        "regime": "BAU",
        "regime_scores": {"BAU": 0.9},
        "regime_check": "FOI",
        "action": "escalate",
        "escalation_reasons": ["Regime check disagrees: classifier BAU, check FOI."],
    }
    pipeline = pipeline_view(values, next_after_finish(values), finished=True)
    summaries = {step["node"]: (step["status"], step["summary"]) for step in pipeline}

    assert summaries["check_regime"] == ("done", "FOI · disagrees")
    assert summaries["compute_deadline"] == ("done", "no deadline")
    assert summaries["route_non_foi"] == ("done", "escalate · 1 reason")
    assert summaries["decide_action"][0] == "skipped"


def test_an_assessed_case_did_not_go_through_route_non_foi():
    values = {
        "request": "x",
        "regime": "FOI",
        "regime_check": "FOI",
        "deadline": "2026-11-23",
        "evidence": [],
        "assessment": SIGNPOST,
        "action": "send",
        "draft": "Fake draft reply.",
        "outcome": "sent",
    }
    steps = statuses(pipeline_view(values, (), finished=True))

    assert steps["route_non_foi"] == steps["human_review"] == "skipped"
    assert steps["decide_action"] == steps["draft_reply"] == steps["log_and_send"] == "done"


def test_sse_puts_each_line_of_data_on_its_own_field():
    assert sse("done", "<p>\n  hi\n</p>") == "event: done\ndata: <p>\ndata:   hi\ndata: </p>\n\n"


def test_review_feedback_is_recorded_on_the_triage_run(monkeypatch):
    calls = []

    class FakeClient:
        def create_feedback(self, run_id, key, **kwargs):
            calls.append((run_id, key, kwargs))

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(cases_module, "Client", FakeClient)
    cases_module.record_review_feedback("run-1", "edit")

    assert calls == [
        ("run-1", "officer_review", {"score": 1, "value": "edit"}),
        ("run-1", "draft_edited", {"score": 1}),
    ]


def test_review_feedback_failure_is_swallowed(monkeypatch):
    class BrokenClient:
        def create_feedback(self, *args, **kwargs):
            raise ConnectionError("LangSmith is down")

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(cases_module, "Client", BrokenClient)
    cases_module.record_review_feedback("run-1", "reject")  # does not raise


# --- Routes -----------------------------------------------------------------------------------


@pytest.fixture
def web(monkeypatch, tmp_path):
    """A function that builds the app over fake model nodes returning the given outputs. The
    regime check always agrees with the classifier.

    The default graph is built on first use, after the fakes are in place, so these tests also
    exercise the SqliteSaver wiring.
    """
    monkeypatch.setenv("FOI_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    classify_calls = []

    def build(regime, fake_assessment=None):
        def classify(state):
            classify_calls.append(state["request"])
            return {"regime": regime, "regime_scores": {regime: 0.9}}

        monkeypatch.setattr(nodes, "classify_regime", classify)
        monkeypatch.setattr(
            nodes,
            "check_regime",
            lambda state: {"regime_check": regime, "regime_check_reason": "Fake check."},
        )
        monkeypatch.setattr(nodes, "assess", lambda state: {"assessment": fake_assessment})
        monkeypatch.setattr(nodes, "draft_reply", lambda state: {"draft": "Fake draft reply."})
        app = create_app(data_dir=tmp_path)
        app.state.classify_calls = classify_calls
        return TestClient(app)

    return build


def start_case(client, request, received="2026-10-26"):
    response = client.post("/triage", data={"request": request, "received": received})
    assert response.status_code == 200
    return re.search(r'sse-connect="/cases/([0-9a-f-]+)/stream"', response.text).group(1)


def events(response):
    """The (event, data) pairs in a text/event-stream body."""
    parsed = []
    for block in response.text.strip().split("\n\n"):
        lines = block.split("\n")
        event = lines[0].removeprefix("event: ")
        data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
        parsed.append((event, data))
    return parsed


def run_case(client, request, received="2026-10-26"):
    case_id = start_case(client, request, received)
    response = client.get(f"/cases/{case_id}/stream")
    assert response.headers["content-type"].startswith("text/event-stream")
    return case_id, events(response)


def test_triage_page_lists_the_examples(web):
    client = web("SAR")
    response = client.get("/")

    assert response.status_code == 200
    examples = client.app.state.cases.examples()
    assert len(examples) > 1
    for example in examples:
        assert f'hx-get="/examples/{example["line"]}"' in response.text


def test_signpost_streams_the_pipeline_then_sends(web, tmp_path):
    client = web("FOI", SIGNPOST)
    case_id, stream = run_case(client, SALARIES_REQUEST)

    names = [event for event, _ in stream]
    assert names[-1] == "done"
    assert set(names[:-1]) == {"pipeline"} and len(names) > 3
    assert client.app.state.cases.case(case_id)["status"] == "sent"
    assert (tmp_path / "outbox" / f"{case_id}.txt").exists()


def test_sar_is_routed_and_skips_the_foi_nodes(web):
    client = web("SAR")
    case_id = start_case(client, SAR_REQUEST)
    final = list(client.app.state.cases.run(case_id))

    event, last = final[-1]
    assert event == "done" and last["case"]["status"] == "routed"
    steps = statuses(final[-2][1]["pipeline"])
    for node in ("check_regime", "compute_deadline", "route_non_foi", "log_and_send"):
        assert steps[node] == "done"
    for node in ("retrieve_evidence", "assess", "draft_reply", "human_review"):
        assert steps[node] == "skipped"
    assert last["case"]["deadline_display"] == "Thu 26 Nov 2026"  # one calendar month


def escalate(web):
    client = web("FOI", EXEMPTION)
    case_id, _ = run_case(client, SALARIES_REQUEST)
    return client, case_id


def test_escalated_case_waits_in_the_review_queue(web):
    client, case_id = escalate(web)

    case = client.app.state.cases.case(case_id)
    assert case["status"] == "awaiting_review"
    assert case["review_url"] == f"/review?case={case_id}"
    assert f'hx-get="/review/{case_id}"' in client.get("/review").text


def test_approval_resumes_sends_and_leaves_the_queue(web, tmp_path, monkeypatch):
    client, case_id = escalate(web)
    response = client.post(f"/review/{case_id}", data={"action": "approve", "draft": ""})

    assert response.status_code == 200
    assert 'hx-swap-oob="true"' in response.text
    assert client.app.state.cases.case(case_id)["status"] == "sent"
    outbox = tmp_path / "outbox" / f"{case_id}.txt"
    assert outbox.read_text(encoding="utf-8") == "Fake draft reply."
    assert f'hx-get="/review/{case_id}"' not in client.get("/review").text


def test_sending_a_changed_reply_sends_it_as_an_edit(web, tmp_path):
    client, case_id = escalate(web)
    reply = "Dear requester,\r\n\r\nOfficer's reply."  # browsers send textarea lines as CRLF
    client.post(f"/review/{case_id}", data={"action": "approve", "draft": reply})

    outbox = tmp_path / "outbox" / f"{case_id}.txt"
    assert outbox.read_text(encoding="utf-8") == "Dear requester,\n\nOfficer's reply."
    assert client.app.state.cases.case(case_id)["review"]["action"] == "edit"


@pytest.mark.parametrize("action", ["approve", "edit"])
def test_sending_the_draft_unchanged_is_an_approval(web, tmp_path, action):
    client, case_id = escalate(web)
    client.post(f"/review/{case_id}", data={"action": action, "draft": "Fake draft reply.\r\n"})

    assert client.app.state.cases.case(case_id)["review"]["action"] == "approve"
    outbox = tmp_path / "outbox" / f"{case_id}.txt"
    assert outbox.read_text(encoding="utf-8") == "Fake draft reply."


def test_a_case_escalated_before_drafting_cannot_be_sent_empty(web, tmp_path):
    client = web("EIR")  # route_non_foi escalates EIR before any FOI work, so there's no draft
    case_id, _ = run_case(client, "How polluted is the air near the incinerator?")
    assert "No draft." in client.get(f"/review/{case_id}").text

    response = client.post(f"/review/{case_id}", data={"action": "approve", "draft": " \r\n"})

    assert response.status_code == 400
    assert client.app.state.cases.case(case_id)["status"] == "awaiting_review"
    assert not (tmp_path / "outbox").exists()


def test_an_officer_can_write_the_reply_a_case_came_without(web, tmp_path):
    client = web("EIR")
    case_id, _ = run_case(client, "How polluted is the air near the incinerator?")
    client.post(f"/review/{case_id}", data={"action": "approve", "draft": "Officer's EIR reply."})

    assert client.app.state.cases.case(case_id)["review"]["action"] == "edit"
    outbox = tmp_path / "outbox" / f"{case_id}.txt"
    assert outbox.read_text(encoding="utf-8") == "Officer's EIR reply."


def test_rejection_ends_the_case_without_sending(web, tmp_path):
    client, case_id = escalate(web)
    client.post(f"/review/{case_id}", data={"action": "reject"})

    case = client.app.state.cases.case(case_id)
    assert case["status"] == "rejected"
    assert case["review"] == {"action": "reject", "action_label": "Rejected"}
    assert not (tmp_path / "outbox").exists()


def test_recently_decided_lists_officer_decisions_only(web):
    client, rejected = escalate(web)
    client.post(f"/review/{rejected}", data={"action": "reject"})
    run_case(client, SALARIES_REQUEST)  # escalates too, but nobody has decided it yet

    decided = client.app.state.cases.decided()
    assert [(item["case_id"], item["outcome"]) for item in decided] == [(rejected, "rejected")]


def test_example_run_can_be_replayed(web, tmp_path):
    client = web("SAR")
    example = client.app.state.cases.example(1)
    case_id, _ = run_case(client, example["request"], example["received"])

    saved = json.loads((tmp_path / "demo_runs" / "1.json").read_text(encoding="utf-8"))
    assert saved["case_id"] == case_id
    assert saved["state"]["outcome"] == "routed"
    assert client.app.state.cases.example(1)["has_replay"]

    response = client.get("/examples/1/replay")
    assert response.status_code == 200
    assert "sse-connect" not in response.text  # nothing runs again
    assert case_id[:8] in response.text
    assert "One calendar month" in response.text  # the SAR deadline rule


def test_a_case_still_finishes_when_the_browser_stops_listening(web):
    client = web("FOI", EXEMPTION)
    case_id = start_case(client, SALARIES_REQUEST)
    stream = client.app.state.cases.run(case_id)
    next(stream)
    stream.close()  # the browser navigated away after the first event

    deadline = time.monotonic() + 5
    while client.app.state.cases.case(case_id)["status"] == "running":
        assert time.monotonic() < deadline, "the run stopped with its stream"
        time.sleep(0.05)
    assert f'hx-get="/review/{case_id}"' in client.get("/review").text


def test_opening_the_stream_twice_does_not_rerun_the_graph(web):
    client = web("SAR")
    case_id, first = run_case(client, SAR_REQUEST)
    second = events(client.get(f"/cases/{case_id}/stream"))

    assert len(client.app.state.classify_calls) == 1
    assert [event for event, _ in second] == ["pipeline", "done"]
    assert second[-1] == first[-1]
