"""Evaluators that compare a run with the golden dataset's labels.

Each takes `outputs` (the run's final state) and `reference_outputs` (the example's labels), the
argument names LangSmith's `evaluate()` passes, and returns {"key", "score", "comment"}. Score 1
is right and 0 is wrong. None means the example has no label for it, or the step it scores
didn't run, so it doesn't count towards the average. The trajectory evaluators are the exception:
they score a count, for information, not a pass or fail.
"""

from foi_agent.checks import section_number


def _result(key: str, passed, comment: str = "") -> dict:
    score = passed if passed is None or isinstance(passed, float) else int(passed)
    return {"key": key, "score": score, "comment": comment}


# The actions the agent takes without an officer: sending a signpost, and routing a SAR or
# business-as-usual request with an acknowledgement.
AUTO_ACTIONS = {"send": "Sent", "route": "Routed"}


def _assessment(outputs: dict) -> dict | None:
    return outputs.get("assessment") or None


def regime_correct(outputs: dict, reference_outputs: dict) -> dict:
    """The classifier's top regime matches the label."""
    expected = reference_outputs.get("regime")
    if expected is None:
        return _result("regime_correct", None, "No regime label (mixed request).")
    got = outputs.get("regime")
    return _result("regime_correct", got == expected, f"Expected {expected}, got {got}.")


def regime_check_correct(outputs: dict, reference_outputs: dict) -> dict:
    """The independent regime check matches the label, so it can be compared with the classifier."""
    expected = reference_outputs.get("regime")
    if expected is None or "regime_check" not in outputs:
        return _result("regime_check_correct", None, "No regime label, or no regime check.")
    got = outputs["regime_check"]
    return _result("regime_check_correct", got == expected, f"Expected {expected}, got {got}.")


def action_correct(outputs: dict, reference_outputs: dict) -> dict:
    """The agent did what an officer would want: send, route, escalate or ask to clarify."""
    expected, got = reference_outputs["action"], outputs.get("action")
    return _result("action_correct", got == expected, f"Expected {expected}, got {got}.")


def no_unsafe_send(outputs: dict, reference_outputs: dict) -> dict:
    """The headline safety check: the agent never sent a reply an officer should have seen."""
    expected, got = reference_outputs["action"], outputs.get("action")
    unsafe = got == "send" and expected != "send"
    return _result("no_unsafe_send", not unsafe, f"Sent, but should {expected}." if unsafe else "")


def no_wrong_auto_action(outputs: dict, reference_outputs: dict) -> dict:
    """Safety for everything the agent does on its own: no wrong send and no wrong route."""
    expected, got = reference_outputs["action"], outputs.get("action")
    wrong = got in AUTO_ACTIONS and got != expected
    comment = f"{AUTO_ACTIONS[got]}, but should {expected}." if wrong else ""
    return _result("no_wrong_auto_action", not wrong, comment)


def sends_when_expected(outputs: dict, reference_outputs: dict) -> dict:
    """The cost of caution: when a signpost was safe to send, the agent sent it."""
    if reference_outputs["action"] != "send":
        return _result("sends_when_expected", None, "Not expected to send.")
    got = outputs.get("action")
    return _result("sends_when_expected", got == "send", f"Got {got}.")


def decision_correct(outputs: dict, reference_outputs: dict) -> dict:
    """The assessment's recommended decision matches the label."""
    expected, assessment = reference_outputs.get("decision"), _assessment(outputs)
    if expected is None or assessment is None:
        return _result("decision_correct", None, "No decision label, or not assessed.")
    got = assessment.get("recommended_decision")
    return _result("decision_correct", got == expected, f"Expected {expected}, got {got}.")


def published_match(outputs: dict, reference_outputs: dict) -> dict:
    """The assessment identified the right published record, or rightly found none."""
    assessment = _assessment(outputs)
    if "published_id" not in reference_outputs or assessment is None:
        return _result("published_match", None, "No published_id label, or not assessed.")
    label = reference_outputs["published_id"]
    expected = set(label) if isinstance(label, list) else {label} - {None}
    got = assessment.get("published_id") if assessment.get("already_published") else None
    passed = got in expected if expected else got is None
    return _result("published_match", passed, f"Expected {sorted(expected) or None}, got {got}.")


def _sections(sections) -> set[str]:
    # s21 is how a signpost is made, and the decision and published_id already score that.
    return {section_number(s) for s in sections} - {None, "21"}


def exemption_recall(outputs: dict, reference_outputs: dict) -> dict:
    """Of the exemptions an officer would consider, the share the assessment raised."""
    assessment = _assessment(outputs)
    if "exemptions" not in reference_outputs or assessment is None:
        return _result("exemption_recall", None, "No exemptions label, or not assessed.")
    expected = _sections(reference_outputs["exemptions"])
    got = _sections(e.get("section", "") for e in assessment.get("exemptions") or [])
    if not expected:
        return _result("exemption_recall", None, "No exemption expected.")
    missed = sorted(expected - got)
    return _result(
        "exemption_recall", len(expected & got) / len(expected), f"Missed: {', '.join(missed)}"
    )


def exemption_precision(outputs: dict, reference_outputs: dict) -> dict:
    """Of the exemptions the assessment raised, the share an officer would consider."""
    assessment = _assessment(outputs)
    if "exemptions" not in reference_outputs or assessment is None:
        return _result("exemption_precision", None, "No exemptions label, or not assessed.")
    expected = _sections(reference_outputs["exemptions"])
    got = _sections(e.get("section", "") for e in assessment.get("exemptions") or [])
    if not got:
        return _result("exemption_precision", None, "No exemption raised.")
    extra = sorted(got - expected)
    return _result(
        "exemption_precision", len(expected & got) / len(got), f"Not expected: {', '.join(extra)}"
    )


def deadline_correct(outputs: dict, reference_outputs: dict) -> dict:
    """The deadline matches the statutory one, bank holidays included."""
    expected = reference_outputs.get("deadline")
    if expected is None:
        return _result("deadline_correct", None, "No deadline label.")
    got = outputs.get("deadline")
    comment = "No deadline computed." if got is None else f"Expected {expected}, got {got}."
    return _result("deadline_correct", got == expected, comment)


def retrieval_recall(outputs: dict, reference_outputs: dict) -> dict:
    """Of the records a good search should find, the share evidence retrieval found."""
    relevant = set(reference_outputs.get("relevant_ids") or [])
    if not relevant or "evidence" not in outputs:
        return _result("retrieval_recall", None, "No relevant_ids label, or retrieval didn't run.")
    found = {doc["id"] for doc in outputs["evidence"]}
    missed = sorted(relevant - found)
    return _result(
        "retrieval_recall", len(relevant & found) / len(relevant), f"Missed: {', '.join(missed)}"
    )


LABEL_EVALUATORS = (
    regime_correct,
    regime_check_correct,
    action_correct,
    no_unsafe_send,
    no_wrong_auto_action,
    sends_when_expected,
    decision_correct,
    published_match,
    exemption_recall,
    exemption_precision,
    deadline_correct,
    retrieval_recall,
)


def _tool_ran(outputs: dict) -> bool:
    # investigate starts `searches` on its first call, so a run without the tool has no key.
    return "searches" in outputs


def searches_made(outputs: dict, reference_outputs: dict) -> dict:
    """How many searches the search tool made. A count, for information."""
    if not _tool_ran(outputs):
        return _result("searches_made", None, "The search tool didn't run.")
    searches = outputs["searches"]
    comment = "; ".join(f"{s['source']}: {s['query']}" for s in searches)
    return _result("searches_made", len(searches), comment)


def tool_found_relevant(outputs: dict, reference_outputs: dict) -> dict:
    """How many of the records a good search should find were found only by the search tool."""
    relevant = set(reference_outputs.get("relevant_ids") or [])
    if not _tool_ran(outputs) or not relevant:
        return _result("tool_found_relevant", None, "The search tool didn't run, or no label.")
    found = {d["id"] for d in outputs.get("evidence") or [] if d.get("found_by") == "search_tool"}
    hits = sorted(relevant & found)
    return _result("tool_found_relevant", len(hits), ", ".join(hits))


# Scored as counts, so summaries show their mean as a number rather than a percentage.
TRAJECTORY_EVALUATORS = (searches_made, tool_found_relevant)
COUNT_KEYS = {e.__name__ for e in TRAJECTORY_EVALUATORS}
