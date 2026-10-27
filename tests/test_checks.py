import pytest

from foi_agent.checks import (
    assessment_consistent,
    citations_grounded,
    draft_links_grounded,
    draft_no_new_personal_data,
    draft_states_rights,
    evidence_found,
    exemption_sections_known,
    run_checks,
    sent_draft_complete,
)

PS_URL = "https://www.northbrook.example/transparency/senior-salaries"
GOOD_DRAFT = (
    f"The salaries are published at {PS_URL}. If you are unhappy, you can ask for an internal "
    "review by emailing foi@northbrook.example, and then complain to the Information "
    "Commissioner's Office at https://ico.org.uk/make-a-complaint/."
)


def foi_run(assessment=None, draft=GOOD_DRAFT, **overrides):
    """A finished FOI run that passes every check unless overridden."""
    base = {
        "request": "Please send me the senior officer salaries for 2025/26.",
        "regime": "FOI",
        "evidence": [
            {"id": "PS-003", "source": "publication_scheme", "url": PS_URL},
            {"id": "EX-S21", "source": "exemption_guidance"},
        ],
        "assessment": {
            "already_published": True,
            "published_id": "PS-003",
            "exemptions": [{"section": "s21", "reason": "Published."}],
            "recommended_decision": "signpost",
            "cited_ids": ["PS-003", "EX-S21"],
            **(assessment or {}),
        },
        "action": "send",
        "draft": draft,
    }
    return {**base, **overrides}


ROUTED_SAR = {
    "request": "Send me everything you hold about me.",
    "regime": "SAR",
    "action": "route",
    "draft": "We have passed your request to our Data Protection team.",
}


def test_a_clean_run_passes_every_check():
    assert [r["score"] for r in run_checks(foi_run())] == [1] * 8


def test_checks_that_need_an_foi_draft_or_assessment_do_not_apply_to_a_routed_sar():
    assert {r["score"] for r in run_checks(ROUTED_SAR)} == {None}


def test_evidence_check_does_not_apply_when_an_unclear_regime_escalated_before_retrieval():
    unclear_foi = {
        "request": "My records, and your policy on housing banding.",
        "regime": "FOI",
        "regime_scores": {"FOI": 0.48, "SAR": 0.45, "EIR": 0.04, "BAU": 0.03},
        "action": "escalate",
        "escalation_reasons": ["Regime unclear: FOI 0.48, SAR 0.45."],
    }
    assert evidence_found(unclear_foi)["score"] is None


def test_evidence_with_only_guidance_fails():
    run = foi_run(evidence=[{"id": "EX-S40", "source": "exemption_guidance"}])
    assert evidence_found(run)["score"] == 0


def test_citing_an_id_that_was_not_retrieved_fails():
    result = citations_grounded(foi_run({"cited_ids": ["PS-003", "DL-2026-099"]}))
    assert result["score"] == 0
    assert "DL-2026-099" in result["comment"]


def test_published_id_that_was_not_retrieved_fails():
    assert citations_grounded(foi_run({"published_id": "PS-099"}))["score"] == 0


@pytest.mark.parametrize(
    "assessment",
    [
        {"published_id": None},
        {"already_published": False, "published_id": None},
        {"recommended_decision": "part_refuse", "exemptions": []},
        {"recommended_decision": "refuse", "exemptions": []},
    ],
)
def test_inconsistent_assessments_fail(assessment):
    assert assessment_consistent(foi_run(assessment))["score"] == 0


@pytest.mark.parametrize("section", ["s40", "Section 43", "s.31", "s40(2)", "s22A", "S12"])
def test_real_foia_sections_are_known(section):
    run = foi_run({"exemptions": [{"section": section, "reason": "x"}]})
    assert exemption_sections_known(run)["score"] == 1


@pytest.mark.parametrize("section", ["s99", "reg 12(5)(e)", "commercial", "s25"])
def test_invented_or_eir_sections_are_flagged(section):
    run = foi_run({"exemptions": [{"section": section, "reason": "x"}]})
    assert exemption_sections_known(run)["score"] == 0


def test_a_link_that_is_not_in_the_evidence_fails():
    draft = GOOD_DRAFT + " See also https://www.northbrook-council.gov.uk/salaries."
    result = draft_links_grounded(foi_run(draft=draft))
    assert result["score"] == 0
    assert "northbrook-council.gov.uk" in result["comment"]


@pytest.mark.parametrize(
    ("draft", "missing"),
    [
        ("You can complain to the Information Commissioner.", "internal review"),
        ("You can ask for an internal review.", "the ICO"),
    ],
)
def test_a_draft_missing_the_requesters_rights_fails(draft, missing):
    result = draft_states_rights(foi_run(draft=draft))
    assert result["score"] == 0
    assert missing in result["comment"]


@pytest.mark.parametrize(
    "leak",
    ["j.smith@gmail.com", "07700 900123", "AB 12 34 56 C", "NB1 4XY"],
)
def test_personal_data_not_in_the_request_fails(leak):
    result = draft_no_new_personal_data(foi_run(draft=f"{GOOD_DRAFT} Contact: {leak}."))
    assert result["score"] == 0


def test_the_icos_postcode_and_helpline_are_allowed():
    # Found on a real traced reply: the ICO's address was flagged as personal data.
    draft = f"{GOOD_DRAFT} Write to Wycliffe House, Wilmslow SK9 5AF, or call 0303 123 1113."
    assert draft_no_new_personal_data(foi_run(draft=draft))["score"] == 1


def test_personal_data_the_requester_gave_is_allowed():
    run = foi_run(
        request="Please reply to j.smith@gmail.com with the salaries.",
        draft=f"{GOOD_DRAFT} We have sent this to j.smith@gmail.com.",
    )
    assert draft_no_new_personal_data(run)["score"] == 1


def test_a_sent_draft_with_ordinary_brackets_and_links_is_complete():
    draft = f"{GOOD_DRAFT} Names are withheld (s40). See [the salaries page]({PS_URL})."
    assert sent_draft_complete(foi_run(draft=draft))["score"] == 1


@pytest.mark.parametrize(
    "gap",
    ["[OFFICER: total cost in 2025/26]", "[officer:number]", "[Insert date of decision]"]
    + ["[number]", "[Amount]", "[DATE]", "[TBC]", "[ number of complaints ]"],
)
def test_a_sent_draft_with_a_placeholder_fails(gap):
    result = sent_draft_complete(foi_run(draft=f"The figure is {gap}. {GOOD_DRAFT}"))
    assert result["score"] == 0
    assert gap in result["comment"]


def test_placeholders_are_expected_in_an_escalated_draft():
    run = foi_run(action="escalate", draft="We spent [OFFICER: total cost] on agency staff.")
    assert sent_draft_complete(run)["score"] is None
