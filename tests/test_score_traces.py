import uuid
from types import SimpleNamespace

from evals.score_traces import feedback_id, is_scorable, score_run

from foi_agent.checks import checks_version

RUN_ID = uuid.UUID("695a5fcb-2cc5-4be6-84dd-0292649e5073")
ROUTED_SAR = {
    "request": "Send me everything you hold about me.",
    "regime": "SAR",
    "regime_guidance": [{"id": "REG-SAR", "source": "regime_guidance"}],
    "action": "route",
    "draft": "We have passed your request to our Data Protection team.",
}


class RecordingClient:
    def __init__(self):
        self.feedback = []

    def create_feedback(self, run_id, key, **kwargs):
        self.feedback.append({"run_id": run_id, "key": key, **kwargs})


def traced_run(outputs, **overrides):
    return SimpleNamespace(
        **{"id": RUN_ID, "outputs": outputs, "error": None, "tags": []} | overrides
    )


def test_rescoring_the_same_run_and_check_reuses_the_feedback_id():
    assert feedback_id(RUN_ID, "evidence_found") == feedback_id(str(RUN_ID), "evidence_found")
    assert feedback_id(RUN_ID, "evidence_found") != feedback_id(RUN_ID, "citations_grounded")


def test_only_finished_case_runs_are_scored():
    assert is_scorable(traced_run(ROUTED_SAR))
    assert not is_scorable(traced_run(ROUTED_SAR, error="Quota exceeded"))
    assert not is_scorable(traced_run({"messages": []}))
    assert not is_scorable(traced_run(ROUTED_SAR, tags=["tracing-check"]))


def test_checks_that_do_not_apply_are_not_written():
    client = RecordingClient()
    score_run(client, traced_run(ROUTED_SAR))

    # A routed SAR has guidance to check, but no assessment or FOI draft.
    assert [f["key"] for f in client.feedback] == ["regime_guidance_found"]
    [written] = client.feedback
    assert written["score"] == 1
    assert written["feedback_id"] == feedback_id(RUN_ID, "regime_guidance_found")
    assert written["source_info"]["checks_version"] == checks_version()


def test_a_dry_run_writes_nothing():
    client = RecordingClient()
    results = score_run(client, traced_run(ROUTED_SAR), dry_run=True)
    assert results and not client.feedback
