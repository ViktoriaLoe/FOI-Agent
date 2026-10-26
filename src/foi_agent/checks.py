"""Code checks over a finished run. They need no labels, so they can score any run.

Each check takes the run's final state as `outputs`, the argument name LangSmith passes to custom
evaluators, so the same functions work in `evaluate()` and on traced runs. Each returns
{"key", "score", "comment"}, with score 1 for pass, 0 for fail and None when the check doesn't
apply to that run (for example, draft checks on a routed SAR).
"""

import hashlib
import re
from functools import cache
from pathlib import Path

from foi_agent.rules import PUBLISHED_SOURCES

# FOIA Part II exemptions plus the s12 cost limit and s14 vexatious or repeated requests.
FOIA_SECTIONS = frozenset(
    {"12", "14", "21", "22", "22A", "23", "24", "26", "27", "28", "29", "30", "31", "32", "33"}
    | {"34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44"}
)

# Addresses a reply may give without being grounded in the evidence: the council's own and the
# ICO's, for internal reviews and complaints.
OWN_DOMAINS = ("northbrook.example", "ico.org.uk")
# The ICO's published postcode (Wilmslow) and helpline, which replies give for complaints.
OWN_CONTACT_DETAILS = frozenset({"SK95AF", "03031231113"})

_URL = re.compile(r"https?://[^\s)>\]\"']+")
_PERSONAL_DATA = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "phone number": re.compile(r"(?:\+44\s?|\b0)(?:\d[\s-]?){9,10}\b"),
    "National Insurance number": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?(?:\d{2}\s?){3}[A-D]\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
}


def _result(key: str, passed: bool | None, comment: str = "") -> dict:
    return {"key": key, "score": None if passed is None else int(passed), "comment": comment}


def _is_foi_assessed(outputs: dict) -> bool:
    return outputs.get("regime") == "FOI" and bool(outputs.get("assessment"))


def _is_own_domain(url: str) -> bool:
    host = url.split("/")[2].lower()
    return any(host == d or host.endswith("." + d) for d in OWN_DOMAINS)


def regime_guidance_found(outputs: dict) -> dict:
    """Retrieval: the classifier was given at least some guidance to work from."""
    ids = [doc["id"] for doc in outputs.get("regime_guidance") or []]
    return _result("regime_guidance_found", bool(ids), ", ".join(ids) or "No guidance retrieved.")


def evidence_found(outputs: dict) -> dict:
    """Retrieval: an FOI case found at least one published record or past response."""
    # Only FOI cases reach retrieval, and not those escalated first because the regime was unclear.
    if "evidence" not in outputs:
        return _result("evidence_found", None, "Evidence retrieval did not run.")
    ids = [d["id"] for d in outputs.get("evidence") or [] if d["source"] in PUBLISHED_SOURCES]
    return _result("evidence_found", bool(ids), ", ".join(ids) or "No published records found.")


def citations_grounded(outputs: dict) -> dict:
    """Assess: every id the assessment relies on was actually retrieved."""
    if not _is_foi_assessed(outputs):
        return _result("citations_grounded", None, "No assessment.")
    assessment = outputs["assessment"]
    cited = set(assessment.get("cited_ids") or []) | {assessment.get("published_id")} - {None}
    retrieved = {doc["id"] for doc in outputs.get("evidence") or []}
    unknown = sorted(cited - retrieved)
    return _result("citations_grounded", not unknown, f"Not retrieved: {', '.join(unknown)}")


def assessment_consistent(outputs: dict) -> dict:
    """Assess: the decision agrees with the fields it rests on."""
    if not _is_foi_assessed(outputs):
        return _result("assessment_consistent", None, "No assessment.")
    a = outputs["assessment"]
    decision = a.get("recommended_decision")
    problems = []
    if a.get("already_published") and not a.get("published_id"):
        problems.append("Already published, but no published_id.")
    if decision == "signpost" and not a.get("already_published"):
        problems.append("Signpost, but not marked as already published.")
    if decision in ("refuse", "part_refuse") and not a.get("exemptions"):
        problems.append(f"Decision {decision!r} with no exemption.")
    return _result("assessment_consistent", not problems, " ".join(problems))


def exemption_sections_known(outputs: dict) -> dict:
    """Assess: every exemption cited is a real FOIA section, not an invented or EIR one."""
    if not _is_foi_assessed(outputs):
        return _result("exemption_sections_known", None, "No assessment.")
    unknown = []
    for exemption in outputs["assessment"].get("exemptions") or []:
        section = exemption.get("section", "")
        match = re.fullmatch(
            r"(?:s|section)\s*\.?\s*(\d+[A-Z]?)(\(.+\))?", section.strip(), re.IGNORECASE
        )
        if not match or match.group(1).upper() not in FOIA_SECTIONS:
            unknown.append(section)
    return _result("exemption_sections_known", not unknown, f"Unrecognised: {', '.join(unknown)}")


def _foi_draft(outputs: dict) -> str | None:
    """The drafted FOI reply, or None for routed acknowledgements and empty drafts."""
    if outputs.get("action") == "route" or not outputs.get("draft"):
        return None
    return outputs["draft"]


def draft_links_grounded(outputs: dict) -> dict:
    """Draft: every link in the reply comes from the evidence, the council or the ICO."""
    draft = _foi_draft(outputs)
    if draft is None:
        return _result("draft_links_grounded", None, "No FOI draft.")
    evidence_urls = {doc.get("url") for doc in outputs.get("evidence") or []}
    links = {url.rstrip(".,;:") for url in _URL.findall(draft)}
    invented = sorted(u for u in links if u not in evidence_urls and not _is_own_domain(u))
    return _result("draft_links_grounded", not invented, f"Not in evidence: {', '.join(invented)}")


def draft_states_rights(outputs: dict) -> dict:
    """Draft: the reply tells the requester about internal review and the ICO."""
    draft = _foi_draft(outputs)
    if draft is None:
        return _result("draft_states_rights", None, "No FOI draft.")
    text = draft.lower()
    missing = []
    if "internal review" not in text:
        missing.append("internal review")
    if "information commissioner" not in text and "ico" not in re.findall(r"\b\w+\b", text):
        missing.append("the ICO")
    return _result("draft_states_rights", not missing, f"Missing: {', '.join(missing)}")


def draft_no_new_personal_data(outputs: dict) -> dict:
    """Draft: the reply contains no contact details or identifiers that weren't in the request.

    The council's and the ICO's own contact details are allowed.
    """
    draft = _foi_draft(outputs)
    if draft is None:
        return _result("draft_no_new_personal_data", None, "No FOI draft.")
    request = outputs.get("request") or ""
    found = []
    for kind, pattern in _PERSONAL_DATA.items():
        for value in pattern.findall(draft):
            own_email = kind == "email address" and value.lower().endswith(OWN_DOMAINS)
            own_contact = re.sub(r"[\s-]", "", value).upper() in OWN_CONTACT_DETAILS
            if value not in request and not (own_email or own_contact):
                found.append(f"{kind} {value!r}")
    return _result("draft_no_new_personal_data", not found, "; ".join(found))


ALL_CHECKS = (
    regime_guidance_found,
    evidence_found,
    citations_grounded,
    assessment_consistent,
    exemption_sections_known,
    draft_links_grounded,
    draft_states_rights,
    draft_no_new_personal_data,
)


@cache
def checks_version() -> str:
    """Short hash of this module. It's recorded with every score, so when a check changes, its
    new scores can be told apart from the old ones."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def run_checks(outputs: dict) -> list[dict]:
    """Every check's result for one run."""
    return [check(outputs) for check in ALL_CHECKS]
