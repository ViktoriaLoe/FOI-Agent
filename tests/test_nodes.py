"""The judgement nodes with their models replaced by fakes (no network), and the deadline node."""

import pytest
from langchain_typesafe import ChoiceAnswer, ClassifierResponse

from foi_agent import nodes
from foi_agent.prompts import CHECK_REGIME_SYSTEM, REGIME_CRITERIA
from foi_agent.state import RegimeCheck

SCORES = {"SAR": 0.82, "FOI": 0.12, "EIR": 0.04, "BAU": 0.02}
GUIDANCE_IDS = ("REG-FOI", "REG-EIR", "REG-SAR", "REG-BAU", "REG-MIXED")


class FakeClassifier:
    """Records the request it is sent and answers with a fixed SAR distribution."""

    def __init__(self):
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        answer = ChoiceAnswer(type="choice", choice="SAR", probabilities=SCORES, confidence=0.7)
        return ClassifierResponse(model="fake", answers={"regime": answer})


def test_classify_regime_stores_the_label_and_distribution(monkeypatch):
    fake = FakeClassifier()
    monkeypatch.setattr(nodes, "_classifier", lambda: fake)
    request = "Under FOI, send me everything you hold about me."

    result = nodes.classify_regime({"request": request})

    assert result == {"regime": "SAR", "regime_scores": SCORES}
    [sent] = fake.requests
    assert sent["state"]["request"] == request
    # Every regime's guidance, every time: a classifier must be able to weigh each option.
    for regime_id in GUIDANCE_IDS:
        assert f"[{regime_id}]" in sent["state"]["ico_guidance"]
    assert set(sent["questions"]) == {"regime"}
    assert sent["questions"]["regime"].criteria == REGIME_CRITERIA
    assert set(REGIME_CRITERIA) == {"FOI", "EIR", "SAR", "BAU"}


class FakeChatModel:
    """Records the schema and messages it is sent and returns a fixed structured answer."""

    def __init__(self, answer):
        self.answer = answer
        self.schemas = []
        self.messages = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        return self.answer


def test_check_regime_gives_an_independent_view_with_all_the_guidance(monkeypatch):
    answer = RegimeCheck(reasoning="The directors are staff, not the requester.", regime="FOI")
    fake = FakeChatModel(answer)
    monkeypatch.setattr(nodes, "_model", lambda: fake)
    request = "Please provide the names and salaries of the directors."

    # The classifier's answer is in state, but the check must not be shown it.
    result = nodes.check_regime({"request": request, "regime": "SAR", "regime_scores": SCORES})

    assert result == {
        "regime_check": "FOI",
        "regime_check_reason": "The directors are staff, not the requester.",
    }
    assert fake.schemas == [RegimeCheck]
    [[system, human]] = fake.messages
    assert system.content == CHECK_REGIME_SYSTEM
    assert request in human.content
    for regime_id in GUIDANCE_IDS:
        assert f"[{regime_id}]" in human.content
    assert "SAR" not in human.content.split("ICO guidance:")[0]
    assert "0.82" not in human.content


@pytest.mark.parametrize(
    ("regime", "deadline"),
    [
        ("FOI", "2026-11-23"),  # 20 working days
        ("EIR", "2026-11-23"),  # 20 working days too
        ("SAR", "2026-11-26"),  # one calendar month
    ],
)
def test_compute_deadline_uses_the_classifiers_regime(regime, deadline):
    state = {"request": "x", "received": "2026-10-26", "regime": regime}
    assert nodes.compute_deadline(state) == {"received": "2026-10-26", "deadline": deadline}


def test_business_as_usual_has_no_deadline():
    state = {"request": "x", "received": "2026-10-26", "regime": "BAU"}
    assert nodes.compute_deadline(state) == {"received": "2026-10-26"}


def test_received_defaults_to_today(monkeypatch):
    class FixedDate(nodes.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 26)

    monkeypatch.setattr(nodes, "date", FixedDate)
    assert nodes.compute_deadline({"request": "x", "regime": "BAU"}) == {"received": "2026-10-26"}
