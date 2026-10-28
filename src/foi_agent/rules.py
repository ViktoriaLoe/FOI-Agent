"""Deterministic routing and the escalate-or-send rule.

The model proposes and this code decides. Nothing here calls an LLM.
"""

import re
from typing import Literal

from foi_agent.language import is_probably_english
from foi_agent.state import FOIState

SIGN_OFF = "Information Governance Team, Northbrook Borough Council"

_ROUTED_REPLIES = {
    "SAR": (
        "Thank you for your request.\n\n"
        "Because you have asked for your own personal information, we are handling your request "
        "as a subject access request under data protection law, not under the Freedom of "
        "Information Act. We have passed it to our Data Protection team, who will respond within "
        "one calendar month. They may contact you first to confirm your identity.\n\n"
        f"Yours sincerely,\n\n{SIGN_OFF}"
    ),
    "BAU": (
        "Thank you for getting in touch.\n\n"
        "Your message is an enquiry about our services rather than a request for recorded "
        "information under the Freedom of Information Act, so we have passed it to Customer "
        "Services, who will reply to you directly.\n\n"
        f"Yours sincerely,\n\n{SIGN_OFF}"
    ),
}

# Only these sources record information the council has actually published.
PUBLISHED_SOURCES = {"disclosure_log", "publication_scheme"}

# Below this probability for the top regime, the regime is unclear and the case escalates.
# A placeholder until it is calibrated against labelled examples on the classifier model in use.
REGIME_MIN_PROB = 0.6

# The most searches the search tool may make on one case, however many the model asks for.
MAX_SEARCHES = 3


def _regime_unclear(state: FOIState) -> bool:
    """True if the classifier's top regime is below REGIME_MIN_PROB.

    With no scores (a caller that sets only `regime`), there is nothing to doubt.
    """
    scores = state.get("regime_scores")
    return bool(scores) and max(scores.values()) < REGIME_MIN_PROB


def _unclear_reason(state: FOIState) -> str:
    """Name the two most likely regimes, as in "Regime unclear: FOI 0.48, SAR 0.41"."""
    top = sorted(state["regime_scores"].items(), key=lambda item: item[1], reverse=True)[:2]
    return "Regime unclear: " + ", ".join(f"{label} {p:.2f}" for label, p in top) + "."


def automatic_action_blockers(state: FOIState) -> list[str]:
    """Why the agent must not act on its own, by sending a reply or routing a case. Empty if
    nothing stands in the way.

    The classifier is confidently wrong too often for its probabilities alone to guard an
    automatic action, so an independent check must agree with it. A case with no check is
    blocked, since there is nothing to confirm the regime.
    """
    blockers = []
    if _regime_unclear(state):
        blockers.append(_unclear_reason(state))
    regime, check = state.get("regime"), state.get("regime_check")
    if check is None:
        blockers.append("No independent regime check was made.")
    elif check != regime:
        blockers.append(f"Regime check disagrees: classifier {regime}, check {check}.")
    if not is_probably_english(state.get("request") or ""):
        blockers.append("The request may not be in English.")
    return blockers


def route_by_regime(state: FOIState) -> Literal["retrieve_evidence", "route_non_foi"]:
    """FOI continues to FOI work only if nothing blocks it. Everything else leaves before it."""
    if state.get("regime") == "FOI" and not automatic_action_blockers(state):
        return "retrieve_evidence"
    return "route_non_foi"


def route_non_foi(state: FOIState) -> dict:
    """Acknowledge SAR and business-as-usual requests if nothing blocks an automatic action.

    Escalate everything else: EIR always, and any case with a blocker, FOI included.
    """
    regime = state.get("regime")
    blockers = automatic_action_blockers(state)
    if regime in _ROUTED_REPLIES and not blockers:
        return {"action": "route", "draft": _ROUTED_REPLIES[regime]}
    if regime == "EIR":
        reasons = ["Environmental information: handle under the EIR."]
    elif regime == "FOI" or regime in _ROUTED_REPLIES:
        reasons = []
    else:
        reasons = [f"Unrecognised regime: {regime!r}."]
    return {"action": "escalate", "escalation_reasons": reasons + blockers}


def decide_action(state: FOIState) -> dict:
    """Send only a clean signpost to information the council has already released, when
    nothing blocks an automatic action.

    Anything else escalates to an FOI officer, with one reason for each condition that failed.
    """
    assessment = state.get("assessment") or {}
    exemptions = assessment.get("exemptions") or []
    decision = assessment.get("recommended_decision")
    if decision == "signpost":
        # Signposting is how s21 is applied, so it doesn't hold the reply back.
        exemptions = [e for e in exemptions if not _is_s21(e.get("section", ""))]
    reasons = []

    if decision != "signpost":
        reasons.append(
            f"Recommended decision is {decision!r}, not a signpost to published information."
        )
    if exemptions:
        sections = ", ".join(e.get("section", "?") for e in exemptions)
        reasons.append(f"Possible exemptions to consider: {sections}.")

    if decision == "signpost":
        published_id = assessment.get("published_id")
        evidence = {doc["id"]: doc for doc in state.get("evidence") or []}
        doc = evidence.get(published_id)
        if not (assessment.get("already_published") and published_id):
            reasons.append("The assessment did not identify where the information is published.")
        elif doc is None:
            reasons.append(f"Published source {published_id} is not among the retrieved evidence.")
        elif doc["source"] not in PUBLISHED_SOURCES:
            reasons.append(
                f"Published source {published_id} is guidance, not a published dataset or response."
            )
        elif doc["source"] == "disclosure_log" and doc.get("outcome") != "released":
            reasons.append(
                f"Disclosure log entry {published_id} was {doc.get('outcome')!r}, "
                "not released in full."
            )

    reasons += automatic_action_blockers(state)
    return {"action": "escalate" if reasons else "send", "escalation_reasons": reasons}


def _is_s21(section: str) -> bool:
    """True for "s21", "s.21", "Section 21" or "s21(1)"."""
    return re.fullmatch(r"(s|section)\s*\.?\s*21(\(\d+\))?", section.strip().lower()) is not None


def route_by_action(state: FOIState) -> Literal["human_review", "log_and_send"]:
    """Escalated cases go to an officer. Sent and routed cases go straight out."""
    return "human_review" if state.get("action") == "escalate" else "log_and_send"


def searches_left(state: FOIState) -> int:
    """How many more searches the search tool may make on this case."""
    return max(MAX_SEARCHES - len(state.get("searches") or []), 0)


def route_after_investigate(state: FOIState) -> Literal["run_searches", "assess"]:
    """Run the searches the model asked for while the budget lasts. Otherwise assess."""
    messages = state.get("investigation") or []
    asked = bool(messages) and bool(getattr(messages[-1], "tool_calls", None))
    return "run_searches" if asked and searches_left(state) else "assess"
