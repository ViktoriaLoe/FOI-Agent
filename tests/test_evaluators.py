from types import SimpleNamespace

import pytest
from evals.evaluators import (
    action_correct,
    deadline_correct,
    decision_correct,
    exemption_precision,
    exemption_recall,
    no_unsafe_send,
    no_wrong_auto_action,
    published_match,
    regime_check_correct,
    regime_correct,
    retrieval_recall,
    sends_when_expected,
)
from evals.judge import JUDGE_VERSION, draft_judge
from evals.run_experiment import evaluator_versions, full_evaluators, summarise
from evals.upload_dataset import load_examples


def assessed(**assessment):
    base = {
        "already_published": False,
        "published_id": None,
        "exemptions": [],
        "recommended_decision": "provide",
    }
    return {"regime": "FOI", "action": "escalate", "assessment": base | assessment}


def score(evaluator, outputs, reference):
    return evaluator(outputs, reference)["score"]


def test_regime_is_compared_with_the_label_and_skipped_for_mixed_requests():
    assert score(regime_correct, {"regime": "SAR"}, {"regime": "SAR"}) == 1
    assert score(regime_correct, {"regime": "FOI"}, {"regime": "SAR"}) == 0
    assert score(regime_correct, {"regime": "FOI"}, {"action": "escalate"}) is None


def test_v1_cannot_clarify_so_clarify_labels_fail_action():
    assert score(action_correct, {"action": "escalate"}, {"action": "clarify"}) == 0


@pytest.mark.parametrize(
    ("got", "expected", "safe"), [("send", "escalate", 0), ("send", "send", 1)]
)
def test_sending_is_only_safe_when_the_label_says_send(got, expected, safe):
    assert score(no_unsafe_send, {"action": got}, {"action": expected}) == safe


def test_escalating_is_never_unsafe():
    assert score(no_unsafe_send, {"action": "escalate"}, {"action": "send"}) == 1


@pytest.mark.parametrize(
    ("got", "expected", "comment"),
    [
        ("route", "escalate", "Routed, but should escalate."),
        ("send", "escalate", "Sent, but should escalate."),
        ("route", "send", "Routed, but should send."),
    ],
)
def test_any_wrong_automatic_action_fails(got, expected, comment):
    result = no_wrong_auto_action({"action": got}, {"action": expected})
    assert result["score"] == 0
    assert result["comment"] == comment


@pytest.mark.parametrize("label", ["route", "send"])
def test_a_correct_automatic_action_passes(label):
    assert score(no_wrong_auto_action, {"action": label}, {"action": label}) == 1


@pytest.mark.parametrize("label", ["send", "route", "clarify"])
def test_escalating_is_never_a_wrong_automatic_action(label):
    assert score(no_wrong_auto_action, {"action": "escalate"}, {"action": label}) == 1


def test_escalating_a_safe_signpost_counts_as_a_missed_send():
    assert score(sends_when_expected, {"action": "escalate"}, {"action": "send"}) == 0
    assert score(sends_when_expected, {"action": "send"}, {"action": "send"}) == 1
    assert score(sends_when_expected, {"action": "escalate"}, {"action": "escalate"}) is None


def test_decision_is_skipped_when_the_case_was_never_assessed():
    reference = {"decision": "refuse"}
    assert score(decision_correct, assessed(recommended_decision="refuse"), reference) == 1
    assert score(decision_correct, assessed(), reference) == 0
    assert score(decision_correct, {"regime": "FOI", "action": "escalate"}, reference) is None


def test_published_match_accepts_any_listed_record_and_requires_none_when_labelled_null():
    signpost = assessed(already_published=True, published_id="DL-2025-007")
    assert score(published_match, signpost, {"published_id": ["PS-009", "DL-2025-007"]}) == 1
    assert score(published_match, signpost, {"published_id": None}) == 0
    assert score(published_match, assessed(), {"published_id": None}) == 1


def test_a_published_id_without_already_published_does_not_count():
    outputs = assessed(already_published=False, published_id="PS-003")
    assert score(published_match, outputs, {"published_id": "PS-003"}) == 0


def test_exemptions_ignore_s21_and_compare_section_numbers():
    outputs = assessed(exemptions=[{"section": "Section 40(2)"}, {"section": "s21"}])
    reference = {"exemptions": ["s21", "s40", "s43"]}
    assert score(exemption_recall, outputs, reference) == 0.5
    assert score(exemption_precision, outputs, reference) == 1.0


def test_an_unexpected_exemption_lowers_precision():
    outputs = assessed(exemptions=[{"section": "s40"}, {"section": "s31"}])
    assert score(exemption_precision, outputs, {"exemptions": ["s40"]}) == 0.5


def test_missing_deadline_fails_when_one_was_due():
    result = deadline_correct({"regime": "EIR"}, {"deadline": "2026-11-23"})
    assert result["score"] == 0
    assert result["comment"] == "No deadline computed."


def test_retrieval_recall_counts_relevant_records_found_and_skips_when_retrieval_did_not_run():
    outputs = {"evidence": [{"id": "PS-003"}, {"id": "EX-S40"}]}
    reference = {"relevant_ids": ["PS-003", "DL-2026-014"]}
    assert score(retrieval_recall, outputs, reference) == 0.5
    assert score(retrieval_recall, {"regime": "SAR"}, reference) is None


def test_upload_ids_are_stable_and_unique():
    first, second = load_examples(), load_examples()
    assert [e["id"] for e in first] == [e["id"] for e in second]
    assert len({e["id"] for e in first}) == len(first)
    assert {e["split"] for e in first} == {"dev", "test"}


def test_summary_averages_only_scores_that_apply():
    def row(slice_, scores):
        results = [SimpleNamespace(key=k, score=v) for k, v in scores.items()]
        example = SimpleNamespace(metadata={"slice": slice_, "category": "edge"})
        return {"example": example, "evaluation_results": {"results": results}}

    rows = [
        row("published", {"no_unsafe_send": 1, "action_correct": 0}),
        row("published", {"no_unsafe_send": 0, "action_correct": None}),
    ]
    summary = summarise(rows, "test")
    assert "| `no_unsafe_send` | 50% (n=2) |" in summary
    assert "| `action_correct` | 0% (n=1) |" in summary


def test_rescoring_can_be_limited_to_named_evaluators_and_leaves_out_the_judge():
    assert draft_judge not in full_evaluators(judge=False)
    chosen = full_evaluators(judge=False, only=["no_wrong_auto_action", "sent_draft_complete"])
    assert [e.__name__ for e in chosen] == ["sent_draft_complete", "no_wrong_auto_action"]
    with pytest.raises(SystemExit):
        full_evaluators(judge=False, only=["no_such_check"])


def test_versions_are_recorded_only_for_evaluators_in_use():
    assert evaluator_versions(full_evaluators(judge=True))["judge_version"] == JUDGE_VERSION
    versions = evaluator_versions([regime_correct])
    assert versions == {"checks_version": None, "judge_version": None}


def test_regime_check_is_scored_against_the_label_and_skipped_without_one():
    reference = {"regime": "FOI"}
    assert score(regime_check_correct, {"regime": "SAR", "regime_check": "FOI"}, reference) == 1
    assert score(regime_check_correct, {"regime": "FOI", "regime_check": "SAR"}, reference) == 0
    assert score(regime_check_correct, {"regime": "FOI"}, reference) is None
    assert score(regime_check_correct, {"regime_check": "FOI"}, {"action": "escalate"}) is None
