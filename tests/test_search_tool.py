"""The search tool: its nodes, the bounded loop and the on/off flag, with a fake model (no network)."""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from foi_agent import nodes
from foi_agent.graph import build_graph, run_metadata
from foi_agent.prompts import INVESTIGATE_SYSTEM
from foi_agent.rules import MAX_SEARCHES

ALLOWANCES_REQUEST = "How much did each councillor claim in allowances and expenses in 2025/26?"
PS_003 = {"source": "publication_scheme", "id": "PS-003", "title": "Senior officer remuneration"}
EX_S40 = {"source": "exemption_guidance", "id": "EX-S40", "title": "Personal information"}


def ask(*searches) -> AIMessage:
    """A model message asking for these (source, query) searches."""
    calls = [
        {"name": "search_records", "args": {"source": s, "query": q}, "id": f"call-{uuid.uuid4()}"}
        for s, q in searches
    ]
    return AIMessage("", tool_calls=calls)


class FakeToolModel:
    """Answers each call with the next scripted message, and records the tools and messages."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.tools = []
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def fake_search(hits_by_query):
    """A stand-in for retrieval.search that returns the given hits for each query."""

    def search(source, query, k=3):
        return [{"source": source, "score": 1.0, **hit} for hit in hits_by_query.get(query, [])]

    return search


def test_investigate_shows_the_model_the_request_and_first_evidence(monkeypatch):
    response = ask(("publication_scheme", "councillor allowances"))
    fake = FakeToolModel(response)
    monkeypatch.setattr(nodes, "_model", lambda: fake)
    state = {"request": ALLOWANCES_REQUEST, "received": "2026-10-26", "evidence": [PS_003]}

    result = nodes.investigate(state)

    assert fake.tools == [nodes.search_records]
    [[system, human]] = fake.calls
    assert system.content == INVESTIGATE_SYSTEM
    assert ALLOWANCES_REQUEST in human.content
    assert "2026-10-26" in human.content
    assert "[PS-003] Senior officer remuneration" in human.content
    # The first call starts the conversation and the search record.
    assert result == {"investigation": [human, response], "searches": []}


def test_investigate_continues_the_conversation_without_resetting_the_searches(monkeypatch):
    fake = FakeToolModel(AIMessage("The evidence is enough."))
    monkeypatch.setattr(nodes, "_model", lambda: fake)
    history = [HumanMessage("Request"), ask(("disclosure_log", "allowances"))]
    history.append(ToolMessage("[DL-1] ...", tool_call_id=history[1].tool_calls[0]["id"]))
    searches = [{"source": "disclosure_log", "query": "allowances", "ids": ["DL-1"]}]

    result = nodes.investigate({"request": "x", "investigation": history, "searches": searches})

    assert fake.calls[0][1:] == history
    assert result == {"investigation": [AIMessage("The evidence is enough.")]}


def test_investigate_makes_no_model_call_once_the_budget_is_used(monkeypatch):
    fake = FakeToolModel()
    monkeypatch.setattr(nodes, "_model", lambda: fake)
    searches = [{"source": "disclosure_log", "query": "q", "ids": []}] * MAX_SEARCHES

    assert nodes.investigate({"request": "x", "searches": searches}) == {}
    assert fake.calls == []


def test_tool_hits_are_added_once_and_tagged_and_each_search_is_recorded(monkeypatch):
    hits = {
        "councillor allowances": [{"id": "PS-006"}, {"id": "PS-003"}],
        "councillor expenses": [{"id": "PS-006"}, {"id": "DL-2025-004", "outcome": "released"}],
    }
    monkeypatch.setattr(nodes, "search", fake_search(hits))
    message = ask(
        ("publication_scheme", "councillor allowances"),
        ("disclosure_log", "councillor expenses"),
    )
    state = {
        "request": "x",
        "evidence": [PS_003, EX_S40],
        "searches": [],
        "investigation": [message],
    }

    result = nodes.run_searches(state)

    assert [d["id"] for d in result["evidence"]] == ["PS-003", "EX-S40", "PS-006", "DL-2025-004"]
    assert "found_by" not in result["evidence"][0]  # found by the first search
    assert [d.get("found_by") for d in result["evidence"][2:]] == ["search_tool"] * 2
    assert result["searches"] == [
        {
            "source": "publication_scheme",
            "query": "councillor allowances",
            "ids": ["PS-006", "PS-003"],
        },
        {
            "source": "disclosure_log",
            "query": "councillor expenses",
            "ids": ["PS-006", "DL-2025-004"],
        },
    ]
    first, second = result["investigation"]
    assert first.tool_call_id == message.tool_calls[0]["id"]
    assert "[PS-006]" in first.content and "Searches left: 2." in first.content
    assert "Searches left: 1." in second.content


def test_calls_beyond_the_budget_in_one_message_are_refused(monkeypatch):
    searched = []
    monkeypatch.setattr(nodes, "search", lambda source, query, k=3: searched.append(query) or [])
    message = ask(("disclosure_log", "a"), ("disclosure_log", "b"), ("disclosure_log", "c"))
    earlier = [{"source": "disclosure_log", "query": "q", "ids": []}] * (MAX_SEARCHES - 1)
    state = {"request": "x", "evidence": [], "searches": earlier, "investigation": [message]}

    result = nodes.run_searches(state)

    assert searched == ["a"]
    assert len(result["searches"]) == MAX_SEARCHES
    replies = result["investigation"]
    # Every call gets an answer, as the model API requires, but only the first is run.
    assert [r.tool_call_id for r in replies] == [c["id"] for c in message.tool_calls]
    assert ["Search limit reached" in r.content for r in replies] == [False, True, True]


def test_a_search_of_an_unknown_source_fails_and_still_counts(monkeypatch):
    monkeypatch.setattr(nodes, "search", fake_search({}))
    message = ask(("council_minutes", "allowances"))
    state = {"request": "x", "evidence": [], "searches": [], "investigation": [message]}

    result = nodes.run_searches(state)

    [reply] = result["investigation"]
    assert reply.status == "error" and reply.content.startswith("Search failed")
    assert result["searches"] == [{"source": "council_minutes", "query": "allowances", "ids": []}]
    assert result["evidence"] == []


# --- The loop in the graph --------------------------------------------------------------------


def signpost(published_id):
    return {
        "reasoning": "Fake assessment.",
        "already_published": True,
        "published_id": published_id,
        "exemptions": [],
        "recommended_decision": "signpost",
        "cited_ids": [published_id],
    }


@pytest.fixture
def graph_with(monkeypatch, tmp_path):
    """A function that builds the graph with fake regime nodes agreeing on FOI, a first search
    that finds only exemption guidance, and `model` as the search tool's model.

    Records the evidence assess is given in `graph_with.assessed`.
    """
    monkeypatch.setenv("FOI_OUTPUT_DIR", str(tmp_path))
    assessed = []
    regime = {"regime": "FOI", "regime_scores": {"FOI": 0.9}}
    monkeypatch.setattr(nodes, "classify_regime", lambda state: regime)
    check = {"regime_check": "FOI", "regime_check_reason": "Fake check."}
    monkeypatch.setattr(nodes, "check_regime", lambda state: check)
    monkeypatch.setattr(nodes, "retrieve_evidence", lambda state: {"evidence": [EX_S40]})
    monkeypatch.setattr(nodes, "draft_reply", lambda state: {"draft": "Fake draft reply."})

    def build(model, assessment, search_tool=None):
        def assess(state):
            assessed.append(state["evidence"])
            return {"assessment": assessment}

        monkeypatch.setattr(nodes, "_model", lambda: model)
        monkeypatch.setattr(nodes, "assess", assess)
        return build_graph(checkpointer=InMemorySaver(), search_tool=search_tool)

    build.assessed = assessed
    return build


def config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_the_loop_stops_at_the_budget_however_often_the_model_asks(graph_with):
    model = FakeToolModel(*[ask(("disclosure_log", f"query {i}")) for i in range(5)])
    graph = graph_with(model, signpost("PS-006"))

    result = graph.invoke({"request": ALLOWANCES_REQUEST}, config())

    assert len(model.calls) == MAX_SEARCHES  # no call once the budget is used
    assert [s["query"] for s in result["searches"]] == ["query 0", "query 1", "query 2"]
    assert len(graph_with.assessed) == 1  # and the case went on to assess


def test_the_tool_finds_the_published_record_and_the_rule_sends(graph_with):
    # The real BM25 search over the knowledge base: PS-006 is councillor allowances, 2025/26.
    model = FakeToolModel(
        ask(("publication_scheme", "councillor allowances expenses")),
        AIMessage("The evidence is enough."),
    )
    graph = graph_with(model, signpost("PS-006"))

    result = graph.invoke({"request": ALLOWANCES_REQUEST, "received": "2026-10-26"}, config())

    assert len(model.calls) == 2
    [evidence] = graph_with.assessed
    found = {d["id"]: d for d in evidence}
    assert found["PS-006"]["found_by"] == "search_tool"
    assert "found_by" not in found["EX-S40"]
    assert result["action"] == "send"
    assert result["outcome"] == "sent"


def test_without_the_tool_the_same_case_escalates_and_the_model_is_not_called(graph_with):
    model = FakeToolModel()  # would fail if called
    graph = graph_with(model, signpost("PS-006"), search_tool=False)

    result = graph.invoke({"request": ALLOWANCES_REQUEST}, config())

    assert model.calls == []
    assert "searches" not in result
    assert not result.get("investigation")  # a list channel, so present but empty
    assert graph_with.assessed == [[EX_S40]]
    assert result["action"] == "escalate"
    reason = "Published source PS-006 is not among the retrieved evidence."
    assert reason in result["escalation_reasons"]


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, True), ("1", True), ("0", False), ("false", False), ("off", False), (" OFF ", False)],
)
def test_the_search_tool_flag_reads_the_environment(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("FOI_SEARCH_TOOL", raising=False)
    else:
        monkeypatch.setenv("FOI_SEARCH_TOOL", value)

    graph = build_graph()

    assert ("investigate" in graph.get_graph().nodes) is enabled
    assert ("run_searches" in graph.get_graph().nodes) is enabled
    assert graph.config["metadata"]["search_tool"] is enabled
    assert run_metadata()["search_tool"] is enabled
    assert run_metadata()["max_searches"] == MAX_SEARCHES


def test_an_explicit_setting_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("FOI_SEARCH_TOOL", "off")
    graph = build_graph(search_tool=True)

    assert "investigate" in graph.get_graph().nodes
    assert graph.config["metadata"]["search_tool"] is True


def test_prompts_show_tool_hits_like_any_other_evidence():
    # So assess and draft_reply read evidence the same way with and without the tool.
    hit = {
        "source": "publication_scheme",
        "id": "PS-006",
        "title": "Allowances",
        "period": "2025/26",
    }
    assert nodes.format_docs([hit | {"found_by": "search_tool"}]) == nodes.format_docs([hit])
