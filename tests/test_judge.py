from types import SimpleNamespace

from evals import judge
from evals.judge import JUDGE_VERSION, Verdict, draft_judge

EXAMPLE = SimpleNamespace(
    inputs={"request": "How much did the council spend on agency staff in 2025?"},
    outputs={"action": "escalate", "decision": "provide"},
    metadata={"why": "Not published, so an officer must gather the figure."},
)
ESCALATED = {
    "action": "escalate",
    "escalation_reasons": ["Not already published."],
    "assessment": {"recommended_decision": "provide"},
    "evidence": [],
    "draft": "The council spent [OFFICER: total agency spend in 2025] on agency staff.",
}


class FakeJudge:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return Verdict(reasoning="The placeholder is marked.", passes=True)


def test_the_judge_is_told_the_case_status(monkeypatch):
    fake = FakeJudge()
    monkeypatch.setattr(judge, "_judge", lambda: fake)
    result = draft_judge(ESCALATED, EXAMPLE)
    case = fake.messages[1].content
    assert "Case status: Escalated to an FOI officer" in case
    assert "Escalation reasons: Not already published." in case
    assert "Recommended decision: provide" in case
    assert result["score"] == 1
    assert result["comment"] == f"[{JUDGE_VERSION}] The placeholder is marked."


def test_a_sent_draft_is_described_as_sent(monkeypatch):
    fake = FakeJudge()
    monkeypatch.setattr(judge, "_judge", lambda: fake)
    draft_judge(ESCALATED | {"action": "send", "escalation_reasons": []}, EXAMPLE)
    assert "Case status: Sent to the requester" in fake.messages[1].content


def test_routed_acknowledgements_are_not_judged(monkeypatch):
    monkeypatch.setattr(judge, "_judge", lambda: None)  # would fail if called
    assert draft_judge(ESCALATED | {"action": "route"}, EXAMPLE)["score"] is None
