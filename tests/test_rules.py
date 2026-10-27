import pytest
from langchain_core.messages import AIMessage, ToolMessage

from foi_agent.rules import (
    MAX_SEARCHES,
    REGIME_MIN_PROB,
    SIGN_OFF,
    automatic_action_blockers,
    decide_action,
    route_after_investigate,
    route_by_action,
    route_by_regime,
    route_non_foi,
    searches_left,
)

PS_003 = {"source": "publication_scheme", "id": "PS-003", "title": "Senior officer remuneration"}
EX_S40 = {"source": "exemption_guidance", "id": "EX-S40", "title": "Personal information"}
DL_RELEASED = {"source": "disclosure_log", "id": "DL-1", "outcome": "released"}
DL_PART = {"source": "disclosure_log", "id": "DL-2", "outcome": "part released"}
EVIDENCE = [PS_003, EX_S40, DL_RELEASED, DL_PART]

SALARIES_REQUEST = "Please send me the senior officer salaries for 2025/26."
POLISH_REQUEST = (
    "Dzień dobry, ile zarabiał dyrektor generalny rady w roku 2025/26 i czy dostał premię?"
)
NOT_ENGLISH = "The request may not be in English."


def signpost(published_id="PS-003", **overrides):
    assessment = {
        "already_published": True,
        "published_id": published_id,
        "exemptions": [],
        "recommended_decision": "signpost",
        "cited_ids": [published_id],
    }
    return {
        "request": SALARIES_REQUEST,
        "regime": "FOI",
        "regime_check": "FOI",
        "evidence": EVIDENCE,
        "assessment": {**assessment, **overrides},
    }


def classified(regime, scores, check=None, request=SALARIES_REQUEST):
    """State after classify_regime and check_regime. The check agrees unless told otherwise."""
    return {
        "request": request,
        "regime": regime,
        "regime_scores": scores,
        "regime_check": check or regime,
    }


CLEAR_FOI = classified("FOI", {"FOI": 0.91, "EIR": 0.05, "SAR": 0.03, "BAU": 0.01})
UNCLEAR_FOI = classified("FOI", {"FOI": 0.48, "SAR": 0.41, "EIR": 0.06, "BAU": 0.05})
CLEAR_SAR = classified("SAR", {"SAR": 0.88, "FOI": 0.10, "EIR": 0.01, "BAU": 0.01})
UNCLEAR_SAR = classified("SAR", {"SAR": 0.45, "BAU": 0.30, "FOI": 0.20, "EIR": 0.05})
CLEAR_BAU = classified("BAU", {"BAU": 0.95, "FOI": 0.03, "SAR": 0.01, "EIR": 0.01})
CLEAR_EIR = classified("EIR", {"EIR": 0.90, "FOI": 0.08, "SAR": 0.01, "BAU": 0.01})


# --- The blockers -----------------------------------------------------------------------------


def test_nothing_blocks_a_clear_regime_the_check_agrees_with():
    assert automatic_action_blockers(CLEAR_FOI) == []


def test_an_unclear_regime_blocks():
    assert automatic_action_blockers(UNCLEAR_FOI) == ["Regime unclear: FOI 0.48, SAR 0.41."]


def test_a_disagreeing_check_blocks():
    state = classified("SAR", CLEAR_SAR["regime_scores"], check="FOI")
    assert automatic_action_blockers(state) == [
        "Regime check disagrees: classifier SAR, check FOI."
    ]


def test_a_missing_check_blocks():
    state = {**CLEAR_SAR, "regime_check": None}
    assert automatic_action_blockers(state) == ["No independent regime check was made."]


def test_a_request_that_may_not_be_in_english_blocks():
    state = classified("FOI", CLEAR_FOI["regime_scores"], request=POLISH_REQUEST)
    assert automatic_action_blockers(state) == [NOT_ENGLISH]


def test_every_blocker_that_applies_is_listed():
    state = classified("SAR", UNCLEAR_SAR["regime_scores"], check="FOI", request=POLISH_REQUEST)
    assert automatic_action_blockers(state) == [
        "Regime unclear: SAR 0.45, BAU 0.30.",
        "Regime check disagrees: classifier SAR, check FOI.",
        NOT_ENGLISH,
    ]


# One state per blocker, applied to a case that would otherwise act on its own.
def blocked(state, blocker):
    if blocker == "unclear":
        top = state["regime"]
        others = [r for r in ("FOI", "EIR", "SAR", "BAU") if r != top]
        return {**state, "regime_scores": {top: 0.5, others[0]: 0.4, others[1]: 0.1}}
    if blocker == "disagrees":
        return {**state, "regime_check": "EIR"}
    return {**state, "request": POLISH_REQUEST}


BLOCKERS = ["unclear", "disagrees", "not_english"]


# --- Routing by regime ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "branch"),
    [
        (CLEAR_FOI, "retrieve_evidence"),
        (CLEAR_EIR, "route_non_foi"),
        (CLEAR_SAR, "route_non_foi"),
        (CLEAR_BAU, "route_non_foi"),
    ],
)
def test_route_by_regime(state, branch):
    assert route_by_regime(state) == branch


def test_foi_exactly_at_the_threshold_continues():
    state = classified("FOI", {"FOI": REGIME_MIN_PROB, "SAR": 1 - REGIME_MIN_PROB})
    assert route_by_regime(state) == "retrieve_evidence"


@pytest.mark.parametrize("blocker", BLOCKERS)
def test_blocked_foi_leaves_before_foi_work_and_escalates(blocker):
    state = blocked(CLEAR_FOI, blocker)
    assert route_by_regime(state) == "route_non_foi"
    result = route_non_foi(state)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == automatic_action_blockers(state) != []
    assert "draft" not in result


def test_unclear_foi_escalates_with_the_two_likeliest_regimes():
    assert route_non_foi(UNCLEAR_FOI) == {
        "action": "escalate",
        "escalation_reasons": ["Regime unclear: FOI 0.48, SAR 0.41."],
    }


def test_foi_the_check_calls_sar_leaves_before_foi_work():
    state = classified("FOI", CLEAR_FOI["regime_scores"], check="SAR")
    assert route_by_regime(state) == "route_non_foi"
    assert route_non_foi(state)["escalation_reasons"] == [
        "Regime check disagrees: classifier FOI, check SAR."
    ]


# --- Routing SAR and business as usual --------------------------------------------------------


@pytest.mark.parametrize("state", [CLEAR_SAR, CLEAR_BAU])
def test_clear_sar_and_bau_are_routed_when_the_check_agrees(state):
    assert route_non_foi(state)["action"] == "route"


@pytest.mark.parametrize("state", [CLEAR_SAR, CLEAR_BAU])
@pytest.mark.parametrize("blocker", BLOCKERS)
def test_sar_and_bau_escalate_instead_of_routing_when_blocked(state, blocker):
    state = blocked(state, blocker)
    result = route_non_foi(state)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == automatic_action_blockers(state) != []
    assert "draft" not in result


def test_sar_the_check_calls_foi_escalates():
    # The six wrong automatic routes in the baseline were FOI requests routed as SARs.
    state = classified("SAR", CLEAR_SAR["regime_scores"], check="FOI")
    assert route_non_foi(state) == {
        "action": "escalate",
        "escalation_reasons": ["Regime check disagrees: classifier SAR, check FOI."],
    }


def test_unclear_sar_escalates_instead_of_routing():
    result = route_non_foi(UNCLEAR_SAR)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == ["Regime unclear: SAR 0.45, BAU 0.30."]
    assert "draft" not in result


def test_sar_is_routed_to_data_protection():
    result = route_non_foi(CLEAR_SAR)
    assert result["action"] == "route"
    assert "Data Protection team" in result["draft"]
    assert "one calendar month" in result["draft"]
    assert "identity" in result["draft"]
    assert SIGN_OFF in result["draft"]


def test_bau_is_routed_to_customer_services():
    result = route_non_foi(CLEAR_BAU)
    assert result["action"] == "route"
    assert "Customer Services" in result["draft"]
    assert SIGN_OFF in result["draft"]


def test_eir_is_escalated_even_when_the_check_agrees():
    assert route_non_foi(CLEAR_EIR) == {
        "action": "escalate",
        "escalation_reasons": ["Environmental information: handle under the EIR."],
    }


def test_eir_escalation_also_lists_the_blockers():
    state = classified("EIR", CLEAR_EIR["regime_scores"], check="FOI")
    assert route_non_foi(state)["escalation_reasons"] == [
        "Environmental information: handle under the EIR.",
        "Regime check disagrees: classifier EIR, check FOI.",
    ]


def test_unrecognised_regime_escalates():
    result = route_non_foi(classified("OTHER", {"OTHER": 0.9}))
    assert result == {"action": "escalate", "escalation_reasons": ["Unrecognised regime: 'OTHER'."]}


# --- The send rule ----------------------------------------------------------------------------


@pytest.mark.parametrize("published_id", ["PS-003", "DL-1"])
def test_clean_signpost_to_released_information_is_sent(published_id):
    assert decide_action(signpost(published_id)) == {"action": "send", "escalation_reasons": []}


@pytest.mark.parametrize("blocker", BLOCKERS)
def test_a_clean_signpost_is_not_sent_when_blocked(blocker):
    state = blocked(signpost(), blocker)
    result = decide_action(state)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == automatic_action_blockers(state) != []


def test_a_signpost_with_no_regime_check_is_not_sent():
    state = {**signpost(), "regime_check": None}
    assert decide_action(state) == {
        "action": "escalate",
        "escalation_reasons": ["No independent regime check was made."],
    }


def test_blockers_are_added_to_the_other_reasons():
    state = {
        **signpost(exemptions=[{"section": "s43", "reason": "Unit rates"}]),
        "regime_check": "EIR",
    }
    assert decide_action(state)["escalation_reasons"] == [
        "Possible exemptions to consider: s43.",
        "Regime check disagrees: classifier FOI, check EIR.",
    ]


def test_decision_other_than_signpost_escalates_with_one_reason():
    state = signpost(recommended_decision="provide", already_published=False, published_id=None)
    result = decide_action(state)
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 1
    assert "'provide'" in result["escalation_reasons"][0]


def test_decision_other_than_signpost_also_lists_exemptions():
    state = signpost(
        recommended_decision="part_refuse",
        exemptions=[{"section": "s40", "reason": "Junior staff names"}],
    )
    result = decide_action(state)
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 2
    assert "s40" in result["escalation_reasons"][1]


def test_exemptions_escalate_a_signpost():
    state = signpost(exemptions=[{"section": "s43", "reason": "Unit rates"}])
    result = decide_action(state)
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 1
    assert "s43" in result["escalation_reasons"][0]


@pytest.mark.parametrize("section", ["s21", "s.21", "Section 21", "s21(1)"])
def test_s21_does_not_hold_back_a_signpost(section):
    state = signpost(exemptions=[{"section": section, "reason": "Already published"}])
    assert decide_action(state) == {"action": "send", "escalation_reasons": []}


def test_s21_alongside_another_exemption_still_escalates():
    state = signpost(
        exemptions=[
            {"section": "s21", "reason": "Salaries are published"},
            {"section": "s40", "reason": "Junior staff names"},
        ]
    )
    result = decide_action(state)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == ["Possible exemptions to consider: s40."]


def test_published_id_not_retrieved_escalates():
    result = decide_action(signpost("PS-999"))
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == [
        "Published source PS-999 is not among the retrieved evidence."
    ]


def test_published_id_pointing_at_guidance_escalates():
    result = decide_action(signpost("EX-S40"))
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 1
    assert "guidance" in result["escalation_reasons"][0]


def test_part_released_disclosure_log_entry_escalates():
    result = decide_action(signpost("DL-2"))
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 1
    assert "part released" in result["escalation_reasons"][0]


def test_not_already_published_escalates():
    result = decide_action(signpost(already_published=False))
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == [
        "The assessment did not identify where the information is published."
    ]


def test_missing_published_id_escalates():
    result = decide_action(signpost(published_id=None))
    assert result["action"] == "escalate"
    assert len(result["escalation_reasons"]) == 1


@pytest.mark.parametrize(
    ("action", "next_node"),
    [("escalate", "human_review"), ("send", "log_and_send"), ("route", "log_and_send")],
)
def test_route_by_action(action, next_node):
    assert route_by_action({"action": action}) == next_node


ASKS = AIMessage("", tool_calls=[{"name": "search_records", "args": {}, "id": "call-1"}])
DONE = AIMessage("The evidence is enough.")


def made(n):
    return [{"source": "disclosure_log", "query": "q", "ids": []}] * n


def test_searches_left_counts_down_from_the_budget_and_stops_at_zero():
    assert searches_left({}) == MAX_SEARCHES == 3
    assert searches_left({"searches": made(2)}) == 1
    assert searches_left({"searches": made(5)}) == 0


@pytest.mark.parametrize(
    ("messages", "searches", "route"),
    [
        ([ASKS], 0, "run_searches"),
        ([ASKS], MAX_SEARCHES - 1, "run_searches"),
        ([ASKS], MAX_SEARCHES, "assess"),  # the budget is used, however many the model wants
        ([DONE], 0, "assess"),  # the model stopped
        ([ASKS, ToolMessage("Searches left: 0.", tool_call_id="call-1")], MAX_SEARCHES, "assess"),
        ([], 0, "assess"),
    ],
)
def test_searches_run_only_when_asked_for_and_within_the_budget(messages, searches, route):
    state = {"investigation": messages, "searches": made(searches)}
    assert route_after_investigate(state) == route
