"""LLM judge for the draft reply, the one thing in a run that code can't score.

It sees the request, the outcome an FOI officer would want (the example's labels and `why`), the
evidence the agent retrieved and the draft, and gives one pass or fail with its reasoning. The
code checks already cover links, the requester's rights and personal data, so the judge doesn't.
"""

import json
import os
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from foi_agent import nodes

JUDGE_SYSTEM = """\
You review draft replies to Freedom of Information requests written for Northbrook Borough \
Council, a UK local authority. You are given the request, the outcome an experienced FOI officer \
decided was right and why, the evidence the drafter had, and the draft.

The draft passes only if all three hold:
1. Substance: it matches the right outcome. It doesn't release or promise what should be \
withheld, doesn't claim information is published when it isn't, doesn't refuse or withhold \
what should be provided, and deals with every part of the request.
2. Grounding: every fact, figure, date and document it states is supported by the evidence. \
General statements of FOI law and the council's own contact details are fine.
3. Clarity: a member of the public would understand it, and it is polite and professional.

Don't judge formatting or length, or whether it mentions internal review or the ICO. Reason \
first, then decide."""


class Verdict(BaseModel):
    reasoning: str = Field(description="Go through substance, grounding and clarity in turn.")
    passes: bool = Field(description="True only if the draft meets all three.")


def judge_model_name() -> str:
    return os.environ.get("FOI_JUDGE_MODEL", nodes.model_name())


@cache
def _judge():
    return init_chat_model(judge_model_name(), max_retries=2, timeout=60).with_structured_output(
        Verdict
    )


def _case(outputs: dict, example) -> str:
    evidence = nodes.format_docs(outputs.get("evidence") or [])
    return (
        f"Request:\n{example.inputs['request']}\n\n"
        f"Right outcome (labels): {json.dumps(example.outputs)}\n"
        f"Why: {example.metadata.get('why', '')}\n\n"
        f"Evidence the drafter had:\n\n{evidence}\n\n"
        f"Draft reply:\n{outputs['draft']}"
    )


def draft_judge(outputs: dict, example) -> dict:
    """Pass or fail for an FOI draft. Routed acknowledgements and empty drafts aren't judged."""
    if outputs.get("action") == "route" or not outputs.get("draft"):
        return {"key": "draft_judge", "score": None, "comment": "No FOI draft."}
    verdict = _judge().invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(_case(outputs, example))])
    return {"key": "draft_judge", "score": int(verdict.passes), "comment": verdict.reasoning}
