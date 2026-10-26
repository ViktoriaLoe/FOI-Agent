import pytest

from foi_agent.rules import SIGN_OFF, decide_action, route_by_action, route_by_regime, route_non_foi

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
