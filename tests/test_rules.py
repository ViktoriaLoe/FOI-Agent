import pytest

from foi_agent.rules import (
    REGIME_MIN_PROB,
    SIGN_OFF,
    decide_action,
    route_by_action,
    route_by_regime,
    route_non_foi,
)

PS_003 = {"source": "publication_scheme", "id": "PS-003", "title": "Senior officer remuneration"}
EX_S40 = {"source": "exemption_guidance", "id": "EX-S40", "title": "Personal information"}
DL_RELEASED = {"source": "disclosure_log", "id": "DL-1", "outcome": "released"}
DL_PART = {"source": "disclosure_log", "id": "DL-2", "outcome": "part released"}
EVIDENCE = [PS_003, EX_S40, DL_RELEASED, DL_PART]


def signpost(published_id="PS-003", **overrides):
    assessment = {
        "already_published": True,
        "published_id": published_id,
        "exemptions": [],
        "recommended_decision": "signpost",
        "cited_ids": [published_id],
    }
    return {"regime": "FOI", "evidence": EVIDENCE, "assessment": {**assessment, **overrides}}


@pytest.mark.parametrize(
    ("regime", "branch"),
    [
        ("FOI", "compute_deadline"),
        ("EIR", "route_non_foi"),
        ("SAR", "route_non_foi"),
        ("BAU", "route_non_foi"),
    ],
)
def test_route_by_regime(regime, branch):
    assert route_by_regime({"regime": regime}) == branch


def classified(regime, scores):
    """State after classify_regime: the top label and the distribution."""
    return {"regime": regime, "regime_scores": scores}


CLEAR_FOI = classified("FOI", {"FOI": 0.91, "EIR": 0.05, "SAR": 0.03, "BAU": 0.01})
UNCLEAR_FOI = classified("FOI", {"FOI": 0.48, "SAR": 0.41, "EIR": 0.06, "BAU": 0.05})
CLEAR_SAR = classified("SAR", {"SAR": 0.88, "FOI": 0.10, "EIR": 0.01, "BAU": 0.01})
UNCLEAR_SAR = classified("SAR", {"SAR": 0.45, "BAU": 0.30, "FOI": 0.20, "EIR": 0.05})
CLEAR_BAU = classified("BAU", {"BAU": 0.95, "FOI": 0.03, "SAR": 0.01, "EIR": 0.01})


def test_clear_foi_continues_to_foi_work():
    assert route_by_regime(CLEAR_FOI) == "compute_deadline"


def test_foi_exactly_at_the_threshold_continues():
    state = classified("FOI", {"FOI": REGIME_MIN_PROB, "SAR": 1 - REGIME_MIN_PROB})
    assert route_by_regime(state) == "compute_deadline"


def test_unclear_foi_leaves_before_foi_work_and_escalates():
    assert route_by_regime(UNCLEAR_FOI) == "route_non_foi"
    assert route_non_foi(UNCLEAR_FOI) == {
        "action": "escalate",
        "escalation_reasons": ["Regime unclear: FOI 0.48, SAR 0.41."],
    }


@pytest.mark.parametrize("state", [CLEAR_SAR, CLEAR_BAU])
def test_clear_sar_and_bau_are_routed(state):
    assert route_by_regime(state) == "route_non_foi"
    assert route_non_foi(state)["action"] == "route"


def test_unclear_sar_escalates_instead_of_routing():
    result = route_non_foi(UNCLEAR_SAR)
    assert result["action"] == "escalate"
    assert result["escalation_reasons"] == ["Regime unclear: SAR 0.45, BAU 0.30."]
    assert "draft" not in result


def test_sar_is_routed_to_data_protection():
    result = route_non_foi({"regime": "SAR"})
    assert result["action"] == "route"
    assert "Data Protection team" in result["draft"]
    assert "one calendar month" in result["draft"]
    assert "identity" in result["draft"]
    assert SIGN_OFF in result["draft"]


def test_bau_is_routed_to_customer_services():
    result = route_non_foi({"regime": "BAU"})
    assert result["action"] == "route"
    assert "Customer Services" in result["draft"]
    assert SIGN_OFF in result["draft"]


def test_eir_is_escalated():
    assert route_non_foi({"regime": "EIR"}) == {
        "action": "escalate",
        "escalation_reasons": ["Environmental information: handle under the EIR."],
    }


@pytest.mark.parametrize("published_id", ["PS-003", "DL-1"])
def test_clean_signpost_to_released_information_is_sent(published_id):
    assert decide_action(signpost(published_id)) == {"action": "send", "escalation_reasons": []}


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
