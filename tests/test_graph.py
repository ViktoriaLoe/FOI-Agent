"""End-to-end runs of the graph with the three LLM nodes replaced by fakes (no network)."""

import json
import uuid

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from foi_agent import nodes
from foi_agent.graph import build_graph
from foi_agent.prompts import PROMPT_VERSION
from foi_agent.retrieval import corpus_version

SALARIES_REQUEST = "Please send me the senior officer salaries for 2025/26."


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


@pytest.fixture
def run(monkeypatch, tmp_path):
    """A function that builds the graph with fake LLM nodes returning the given outputs."""
    monkeypatch.setenv("FOI_OUTPUT_DIR", str(tmp_path))

    def build(regime, fake_assessment=None):
        monkeypatch.setattr(
            nodes, "classify_regime", lambda state: {"regime": regime, "regime_reason": "Fake."}
        )
        monkeypatch.setattr(nodes, "assess", lambda state: {"assessment": fake_assessment})
        monkeypatch.setattr(nodes, "draft_reply", lambda state: {"draft": "Fake draft reply."})
        return build_graph(checkpointer=InMemorySaver())

    return build


def config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def log_lines(tmp_path):
    lines = (tmp_path / "case_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_sar_is_routed(run, tmp_path):
    graph, cfg = run("SAR"), config()
    result = graph.invoke({"request": "Send me everything you hold about me."}, cfg)

    assert result["outcome"] == "routed"
    assert "__interrupt__" not in result
    assert "deadline" not in result  # left before any FOI work
    outbox_file = tmp_path / "outbox" / f"{cfg['configurable']['thread_id']}.txt"
    assert "Data Protection team" in outbox_file.read_text(encoding="utf-8")
    assert log_lines(tmp_path)[0]["outcome"] == "routed"


def test_foi_signpost_to_retrieved_publication_is_sent(run, tmp_path):
    graph = run(
        "FOI",
        assessment(
            already_published=True,
            published_id="PS-003",
            recommended_decision="signpost",
            cited_ids=["PS-003"],
        ),
    )
    cfg = config()
    result = graph.invoke({"request": SALARIES_REQUEST, "received": "2026-10-26"}, cfg)

    assert "PS-003" in [doc["id"] for doc in result["evidence"]]
    assert "__interrupt__" not in result
    assert result["action"] == "send"
    assert result["deadline"] == "2026-11-23"
    assert result["outcome"] == "sent"
    assert (tmp_path / "outbox" / f"{cfg['configurable']['thread_id']}.txt").exists()


def exemption_graph(run):
    return run(
        "FOI",
        assessment(
            recommended_decision="part_refuse",
            exemptions=[{"section": "s40", "reason": "Junior staff salaries"}],
        ),
    )


def test_foi_with_exemption_waits_for_review_and_sends_on_approval(run, tmp_path):
    graph, cfg = exemption_graph(run), config()
    result = graph.invoke({"request": SALARIES_REQUEST}, cfg)

    payload = result["__interrupt__"][0].value
    assert graph.get_state(cfg).next == ("human_review",)
    assert payload["draft"] == "Fake draft reply."
    assert any("s40" in reason for reason in payload["escalation_reasons"])

    result = graph.invoke(Command(resume="approve"), cfg)
    assert result["outcome"] == "sent"
    assert log_lines(tmp_path)[0]["review_action"] == "approve"


def test_foi_with_exemption_is_not_sent_when_rejected(run, tmp_path):
    graph, cfg = exemption_graph(run), config()
    graph.invoke({"request": SALARIES_REQUEST}, cfg)

    result = graph.invoke(Command(resume={"action": "reject"}), cfg)
    assert result["outcome"] == "rejected"
    assert not (tmp_path / "outbox").exists()


class MetadataRecorder(BaseCallbackHandler):
    """Records the metadata and tags each graph step is traced with."""

    def __init__(self):
        self.steps = {}

    def on_chain_start(self, serialized, inputs, *, metadata=None, tags=None, **kwargs):
        self.steps[kwargs.get("name")] = {"metadata": metadata or {}, "tags": tags or []}


def test_every_step_is_traced_with_run_metadata(run):
    graph, recorder = run("SAR"), MetadataRecorder()
    cfg = {**config(), "callbacks": [recorder], "tags": ["test"]}
    graph.invoke({"request": "Send me everything you hold about me."}, cfg)

    assert "foi_triage" in recorder.steps  # the root run
    for step in ("foi_triage", "classify_regime", "route_non_foi", "log_and_send"):
        metadata = recorder.steps[step]["metadata"]
        assert metadata["prompt_version"] == PROMPT_VERSION
        assert metadata["corpus_version"] == corpus_version()
        assert metadata["model"] == nodes.model_name()
        assert metadata["thread_id"] == cfg["configurable"]["thread_id"]
        assert "test" in recorder.steps[step]["tags"]


def test_eir_waits_for_review(run):
    graph, cfg = run("EIR"), config()
    result = graph.invoke({"request": "Air quality readings near the incinerator."}, cfg)

    assert graph.get_state(cfg).next == ("human_review",)
    payload = result["__interrupt__"][0].value
    assert payload["escalation_reasons"] == ["Environmental information: handle under the EIR."]
