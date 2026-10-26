"""classify_regime with the decision model replaced by a fake (no network)."""

from langchain_typesafe import ChoiceAnswer, ClassifierResponse

from foi_agent import nodes
from foi_agent.prompts import REGIME_CRITERIA

GUIDANCE = [{"id": "REG-SAR", "title": "Right of access", "test": "Their own personal data."}]
SCORES = {"SAR": 0.82, "FOI": 0.12, "EIR": 0.04, "BAU": 0.02}


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

    result = nodes.classify_regime({"request": request, "regime_guidance": GUIDANCE})

    assert result == {"regime": "SAR", "regime_scores": SCORES}
    [sent] = fake.requests
    assert sent["state"]["request"] == request
    assert sent["state"]["ico_guidance"].startswith("[REG-SAR] Right of access")
    assert set(sent["questions"]) == {"regime"}
    assert sent["questions"]["regime"].criteria == REGIME_CRITERIA
    assert set(REGIME_CRITERIA) == {"FOI", "EIR", "SAR", "BAU"}
