"""Graph nodes: retrieval, the LLM judgement nodes, the deadline, officer review and sending.

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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, interrupt

from foi_agent.deadline import foi_deadline
from foi_agent.prompts import ASSESS_SYSTEM, CLASSIFY_SYSTEM, DRAFT_SYSTEM
from foi_agent.retrieval import search
from foi_agent.state import Assessment, FOIState, RegimeDecision

DEFAULT_MODEL = "google_genai:gemini-3.8-flash"
EVIDENCE_SOURCES = ("disclosure_log", "publication_scheme", "exemption_guidance")


@cache
def _model():
    # Created on first use, so importing the graph doesn't need an API key.
    return init_chat_model(os.environ.get("FOI_MODEL", DEFAULT_MODEL), temperature=0)


def format_docs(docs: list[dict]) -> str:
    """Render retrieved records for a prompt: "[id] title", then the other fields."""
    blocks = []
    for doc in docs:
        lines = [f"[{doc['id']}] {doc.get('title', '')}".rstrip()]
        for key, value in doc.items():
            if key in {"id", "title", "source", "score"}:
                continue
            if isinstance(value, list):
                value = "; ".join(map(str, value)) or "none"
            lines.append(f"{key}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "(no records found)"


def retrieve_regime_guidance(state: FOIState) -> dict:
    return {"regime_guidance": search("regime_guidance", state["request"], k=3)}


def classify_regime(state: FOIState) -> dict:
    system = CLASSIFY_SYSTEM.format(guidance=format_docs(state.get("regime_guidance") or []))
    model = _model().with_structured_output(RegimeDecision)
    decision = model.invoke([SystemMessage(system), HumanMessage(state["request"])])
    return {"regime": decision.regime, "regime_reason": decision.reason}


def compute_deadline(state: FOIState) -> dict:
    received = date.fromisoformat(state["received"]) if state.get("received") else date.today()
    return {"received": received.isoformat(), "deadline": foi_deadline(received).isoformat()}


def retrieve_evidence(state: FOIState) -> dict:
    # v1 searches with the raw request text.
    evidence = [doc for source in EVIDENCE_SOURCES for doc in search(source, state["request"], k=3)]
    return {"evidence": evidence}


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
