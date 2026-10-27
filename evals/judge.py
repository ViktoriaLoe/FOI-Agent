"""LLM judge for the draft reply, the one thing in a run that code can't score.

It sees the request, the outcome an FOI officer would want (the example's labels and `why`), the
case status (sent or escalated), the evidence the agent retrieved and the draft, and gives one
pass or fail with its reasoning. The code checks already cover links, the requester's rights,
personal data and placeholders in sent replies, so the judge doesn't.
"""

import json
import os
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from foi_agent import nodes

# j1 judged every draft as a finished letter. j2 judges each draft by its purpose.
JUDGE_VERSION = "j2"

JUDGE_SYSTEM = """\
You review draft replies to Freedom of Information requests written for Northbrook Borough \
Council, a UK local authority. You are given the request, the outcome an experienced FOI officer \
decided was right and why, the case status, the evidence the drafter had, and the draft.

Judge each draft by its purpose, which the case status tells you:
- Sent: the draft went to the requester as it is, so it must be a complete, correct reply.
- Escalated: the draft is a starting point for an FOI officer, who checks and finishes it. \
Clearly marked placeholders for information the officer must supply, such as \
[OFFICER: number of complaints in 2025], are correct and are not a fault. Accept any obvious \
placeholder. Inventing a figure, date or fact that isn't in the evidence is a fail. An \
escalated draft still fails for the wrong substance, an unsupported claim or the wrong legal \
basis, under the criteria below.

The draft passes only if all three hold:
1. Substance: it matches the right outcome. It doesn't release or promise what should be \
withheld, doesn't claim information is published when it isn't, doesn't refuse or withhold \
what should be provided, relies on the right legal basis, and deals with every part of the \
request.
2. Grounding: every fact, figure, date and document it states is supported by the evidence. \
General statements of FOI law and the council's own contact details are fine.
3. Clarity: a member of the public would understand it, and it is polite and professional. \
For an escalated draft, judge the text around the placeholders.

Don't judge formatting or length, or whether it mentions internal review or the ICO. Reason \
first, then decide."""


class Verdict(BaseModel):
    reasoning: str = Field(description="Go through substance, grounding and clarity in turn.")
    passes: bool = Field(description="True only if the draft meets all three.")


# A different model generation from the agent's, so the judge doesn't favour drafts written in
# its own style, and a dated snapshot, so its scores don't drift between experiments.
DEFAULT_JUDGE_MODEL = "openai:gpt-5.5-2026-04-23"


def judge_model_name() -> str:
    return os.environ.get("FOI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


@cache
def _judge():
    return init_chat_model(judge_model_name(), max_retries=2, timeout=60).with_structured_output(
        Verdict
    )


STATUS = {"send": "Sent to the requester", "escalate": "Escalated to an FOI officer"}


def _status(outputs: dict) -> str:
    """What the agent did with the draft, why, and the decision the assessment recommended."""
    action = outputs.get("action")
    reasons = "; ".join(outputs.get("escalation_reasons") or []) or "None"
    decision = (outputs.get("assessment") or {}).get("recommended_decision")
    return (
        f"Case status: {STATUS.get(action, action)}\n"
        f"Escalation reasons: {reasons}\n"
        f"Recommended decision: {decision}"
    )


def _case(outputs: dict, example) -> str:
    evidence = nodes.format_docs(outputs.get("evidence") or [])
    return (
        f"Request:\n{example.inputs['request']}\n\n"
        f"Right outcome (labels): {json.dumps(example.outputs)}\n"
        f"Why: {example.metadata.get('why', '')}\n\n"
        f"{_status(outputs)}\n\n"
        f"Evidence the drafter had:\n\n{evidence}\n\n"
        f"Draft reply:\n{outputs['draft']}"
    )


def draft_judge(outputs: dict, example) -> dict:
    """Pass or fail for an FOI draft. Routed acknowledgements and empty drafts aren't judged."""
    if outputs.get("action") == "route" or not outputs.get("draft"):
        return {"key": "draft_judge", "score": None, "comment": "No FOI draft."}
    verdict = _judge().invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(_case(outputs, example))])
    return {
        "key": "draft_judge",
        "score": int(verdict.passes),
        "comment": f"[{JUDGE_VERSION}] {verdict.reasoning}",
    }
