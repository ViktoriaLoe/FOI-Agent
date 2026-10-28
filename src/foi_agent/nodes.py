"""Graph nodes: the judgement nodes, the deadline, retrieval and the search tool, officer review
and sending.

The deterministic routing and escalation rule live in `foi_agent.rules`.
"""

import json
import os
import uuid
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_typesafe import Choice, TypeSafeClassifier
from langgraph.graph import END
from langgraph.types import Command, interrupt

from foi_agent.deadline import foi_deadline, sar_deadline
from foi_agent.prompts import (
    ASSESS_SYSTEM,
    CHECK_REGIME_SYSTEM,
    DRAFT_SYSTEM,
    INVESTIGATE_SYSTEM,
    REGIME_CRITERIA,
    REGIME_INSTRUCTIONS,
)
from foi_agent.retrieval import load_records, search
from foi_agent.rules import MAX_SEARCHES, searches_left
from foi_agent.state import Assessment, FOIState, RegimeCheck

DEFAULT_MODEL = "openai:gpt-6-luna"
DEFAULT_CLASSIFIER_MODEL = "semif-qwen3.5-4b"
GATEWAY_URL = "https://gateway.smith.langchain.com"
EVIDENCE_SOURCES = ("disclosure_log", "publication_scheme", "exemption_guidance")
RETRIEVAL_K = 3  # records per source
# The deadline rule for each regime the classifier can give. Business as usual has no deadline.
DEADLINE_RULES = {"FOI": foi_deadline, "EIR": foi_deadline, "SAR": sar_deadline}


def model_name() -> str:
    return os.environ.get("FOI_MODEL", DEFAULT_MODEL)


@cache
def _model():
    # Created on first use, so importing the graph doesn't need an API key.
    # Fail within a couple of minutes and let the node's RetryPolicy try again.
    return init_chat_model(model_name(), max_retries=2, timeout=60)


def classifier_model_name() -> str:
    return os.environ.get("FOI_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)


@cache
def _classifier() -> TypeSafeClassifier:
    # The decision model for regime classification, served by the LangSmith Gateway, which takes
    # a LangSmith key. Created on first use, like _model().
    return TypeSafeClassifier(
        model=classifier_model_name(),
        api_key=os.environ["LANGSMITH_API_KEY"],
        base_url=GATEWAY_URL,
    )


def format_docs(docs: list[dict]) -> str:
    """Render retrieved records for a prompt: "[id] title", then the other fields."""
    blocks = []
    for doc in docs:
        lines = [f"[{doc['id']}] {doc.get('title', '')}".rstrip()]
        for key, value in doc.items():
            if key in {"id", "title", "source", "score", "found_by"}:
                continue
            if isinstance(value, list):
                value = "; ".join(map(str, value)) or "none"
            lines.append(f"{key}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "(no records found)"


@cache
def regime_guidance() -> str:
    """The ICO guidance summary for every regime, all of which the classifier sees every time.

    A classifier has to weigh every option, so this is not retrieved: searching it once handed
    the classifier only some of the tests, and regime accuracy fell to 55%.
    """
    return format_docs(load_records("regime_guidance"))


def classify_regime(state: FOIState) -> dict:
    """Ask the decision model which regime applies.

    Keeps the whole distribution, so the routing rule can escalate when no regime is clearly ahead.
    """
    response = _classifier().invoke(
        {
            "state": {
                "request": state["request"],
                "ico_guidance": regime_guidance(),
            },
            "questions": {
                "regime": Choice(instructions=REGIME_INSTRUCTIONS, criteria=REGIME_CRITERIA)
            },
        }
    )
    answer = response.choices["regime"]
    return {"regime": answer.choice, "regime_scores": dict(answer.probabilities)}


def check_regime(state: FOIState) -> dict:
    """A second, independent view of the regime from the main LLM.

    The classifier is confidently wrong too often for its probabilities to guard an automatic
    reply, so the rules act on their own only when this check agrees. It is not shown the
    classifier's answer.
    """
    prompt = f"Request:\n{state['request']}\n\nICO guidance:\n\n{regime_guidance()}"
    model = _model().with_structured_output(RegimeCheck)
    check = model.invoke([SystemMessage(CHECK_REGIME_SYSTEM), HumanMessage(prompt)])
    return {"regime_check": check.regime, "regime_check_reason": check.reasoning}


def compute_deadline(state: FOIState) -> dict:
    """The response deadline under the classifier's regime, before any routing.

    Every case gets one, escalated ones included, except business as usual, which has none.
    """
    received = date.fromisoformat(state["received"]) if state.get("received") else date.today()
    update = {"received": received.isoformat()}
    rule = DEADLINE_RULES.get(state.get("regime"))
    if rule:
        update["deadline"] = rule(received).isoformat()
    return update


def retrieve_evidence(state: FOIState) -> dict:
    # v1 searches with the raw request text.
    evidence = [
        doc
        for source in EVIDENCE_SOURCES
        for doc in search(source, state["request"], k=RETRIEVAL_K)
    ]
    return {"evidence": evidence}


@tool(response_format="content_and_artifact", parse_docstring=True)
def search_records(
    source: Literal["disclosure_log", "publication_scheme", "exemption_guidance"], query: str
) -> tuple[str, list[dict]]:
    """Search one source of the council's records and return the best matches. Read-only.

    disclosure_log holds past FOI responses with their outcomes. publication_scheme holds the
    datasets the council publishes routinely, with the periods they cover. exemption_guidance
    holds ICO guidance on the FOI exemptions.

    Args:
        source: The source to search.
        query: Words to search for, ideally as the council's own records would put them.
    """
    hits = search(source, query, k=RETRIEVAL_K)
    return format_docs(hits), hits


def investigate(state: FOIState) -> dict:
    """Let the model look at the first search's evidence and ask for more searches if needed.

    Makes no model call once the search budget is used up.
    """
    if not searches_left(state):
        return {}
    history = state.get("investigation") or []
    # The first call starts the conversation, and its search record, which later calls continue.
    start = []
    if not history:
        prompt = (
            f"Received: {state.get('received')}\n\n"
            f"Request:\n{state['request']}\n\n"
            f"Evidence from the first search:\n\n{format_docs(state.get('evidence') or [])}"
        )
        start = [HumanMessage(prompt)]
    model = _model().bind_tools([search_records])
    response = model.invoke([SystemMessage(INVESTIGATE_SYSTEM), *history, *start])
    update = {"investigation": [*start, response]}
    if start:
        update["searches"] = []
    return update


def run_searches(state: FOIState) -> dict:
    """Run the searches the last message asked for, up to the budget, and refuse the rest.

    New hits are added to the evidence once each, tagged as found by the search tool.
    """
    evidence = list(state.get("evidence") or [])
    searches = list(state.get("searches") or [])
    seen = {doc["id"] for doc in evidence}
    replies = []
    for call in state["investigation"][-1].tool_calls:
        if len(searches) >= MAX_SEARCHES:
            refusal = "Search limit reached: no more searches can be made on this case."
            replies.append(ToolMessage(refusal, tool_call_id=call["id"], name=call["name"]))
            continue
        try:
            reply = search_records.invoke(call)  # a ToolMessage carrying the hits as its artifact
            hits = reply.artifact
        except ValueError as exc:  # arguments that don't fit the schema, such as an unknown source
            reply = ToolMessage(f"Search failed: {exc}", tool_call_id=call["id"], status="error")
            hits = []
        args = call.get("args") or {}
        ids = [hit["id"] for hit in hits]
        searches.append({"source": args.get("source"), "query": args.get("query"), "ids": ids})
        for hit in hits:
            if hit["id"] not in seen:
                seen.add(hit["id"])
                evidence.append({**hit, "found_by": "search_tool"})
        reply.content = f"{reply.content}\n\nSearches left: {MAX_SEARCHES - len(searches)}."
        replies.append(reply)
    return {"evidence": evidence, "searches": searches, "investigation": replies}


def assess(state: FOIState) -> dict:
    prompt = (
        f"Received: {state.get('received')}\n\n"
        f"Request:\n{state['request']}\n\n"
        f"Evidence:\n\n{format_docs(state.get('evidence') or [])}"
    )
    model = _model().with_structured_output(Assessment)
    assessment = model.invoke([SystemMessage(ASSESS_SYSTEM), HumanMessage(prompt)])
    return {"assessment": assessment.model_dump()}


def draft_reply(state: FOIState) -> dict:
    if state.get("action") == "escalate":
        status = "Escalated to an FOI officer. Reasons: " + " ".join(
            state.get("escalation_reasons") or []
        )
    else:
        status = "Low risk: the reply will be sent as written."
    prompt = (
        f"Case status: {status}\n\n"
        f"Received: {state.get('received')}\n"
        f"Response deadline: {state.get('deadline')}\n\n"
        f"Request:\n{state['request']}\n\n"
        f"Assessment:\n{json.dumps(state.get('assessment'), indent=2)}\n\n"
        f"Evidence:\n\n{format_docs(state.get('evidence') or [])}"
    )
    response = _model().invoke([SystemMessage(DRAFT_SYSTEM), HumanMessage(prompt)])
    return {"draft": response.text}


def human_review(state: FOIState) -> Command[Literal["log_and_send", "__end__"]]:
    """Pause for an FOI officer to approve, edit or reject the case.

    Resume with {"action": "approve" | "edit" | "reject", "draft": optional edited reply}, or a
    bare string such as "approve". Only approve and edit lead to sending; anything else ends the
    case without sending.
    """
    decision = interrupt(
        {
            "regime": state.get("regime"),
            "deadline": state.get("deadline"),
            "escalation_reasons": state.get("escalation_reasons"),
            "assessment": state.get("assessment"),
            "draft": state.get("draft"),
        }
    )
    if isinstance(decision, str):
        decision = {"action": decision}
    action = decision.get("action")
    review = {"action": action}

    if action in ("approve", "edit"):
        update = {"review": review}
        if decision.get("draft"):
            update["draft"] = decision["draft"]
        return Command(update=update, goto="log_and_send")
    return Command(update={"review": review, "outcome": "rejected"}, goto=END)


def log_and_send(state: FOIState, config: RunnableConfig) -> dict:
    """Write the reply to the outbox and append the case to the case log. Nothing is emailed."""
    case_id = (config.get("configurable") or {}).get("thread_id") or str(uuid.uuid4())
    outcome = "routed" if state.get("action") == "route" else "sent"
    out_dir = Path(os.environ.get("FOI_OUTPUT_DIR", "var"))

    outbox = out_dir / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    (outbox / f"{case_id}.txt").write_text(state.get("draft") or "", encoding="utf-8")

    record = {
        "case_id": case_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "regime": state.get("regime"),
        "action": state.get("action"),
        "escalation_reasons": state.get("escalation_reasons") or [],
        "review_action": (state.get("review") or {}).get("action"),
        "outcome": outcome,
    }
    with (out_dir / "case_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
    return {"outcome": outcome}
