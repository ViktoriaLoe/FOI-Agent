"""Graph state and the structured output the LLM nodes return.

State holds plain JSON-serialisable values, so model outputs are stored with `.model_dump()`. The
one exception is `investigation`, the search tool's messages.
"""

from typing import Annotated, Literal, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Regime = Literal["FOI", "EIR", "SAR", "BAU"]
Decision = Literal["provide", "signpost", "part_refuse", "refuse"]
Action = Literal["send", "escalate", "route"]


class Exemption(BaseModel):
    """An FOI exemption that may apply to some or all of the request."""

    section: str = Field(description="The FOIA section, written like 's40' or 's43'.")
    reason: str = Field(description="Why this exemption may apply to this request.")


class Assessment(BaseModel):
    """The assessment of an FOI request against the retrieved evidence."""

    reasoning: str = Field(
        description=(
            "Think this through before deciding: what is being asked for, which evidence "
            "records are relevant, whether they cover the period asked for, and whether any "
            "exemption may apply."
        )
    )
    already_published: bool = Field(
        description=(
            "True only if a publication scheme record or a released disclosure log entry in the "
            "evidence fully answers the request, including the period asked for."
        )
    )
    published_id: str | None = Field(
        description=(
            "The id of the evidence record where the information is published, or null if it "
            "is not already published."
        )
    )
    exemptions: list[Exemption] = Field(
        description="Exemptions that may apply to any part of the request. Empty if none."
    )
    recommended_decision: Decision = Field(
        description=(
            "provide: release the information; signpost: point the requester to where it is "
            "already published; part_refuse: release some and withhold some under an exemption; "
            "refuse: withhold it all or refuse the request."
        )
    )
    cited_ids: list[str] = Field(
        description="Ids of the evidence records relied on. Use only ids shown in the evidence."
    )


class RegimeCheck(BaseModel):
    """An independent second opinion on which access regime applies to a request."""

    reasoning: str = Field(
        description=(
            "Think this through before deciding: who the information is about (the requester "
            "themselves or someone they act for, other people such as staff or councillors, or "
            "nobody in particular), whether it is environmental, and whether the requester wants "
            "recorded information or wants the council to do something."
        )
    )
    regime: Regime = Field(
        description=(
            "SAR only if the requester wants their own personal data or that of someone they act "
            "for; EIR for environmental information, whatever the requester calls it; BAU for a "
            "service request, complaint or routine enquiry; otherwise FOI."
        )
    )


class InputState(TypedDict):
    """What a caller provides to start a case."""

    request: str
    received: NotRequired[str]  # ISO date; defaults to today


class FOIState(InputState, total=False):
    """Full graph state."""

    regime: Regime
    regime_scores: dict[str, float]  # the classifier's probability for each regime
    regime_check: Regime  # the independent check's regime
    regime_check_reason: str
    deadline: str  # ISO date
    evidence: list[dict]  # search tool hits are tagged "found_by": "search_tool"
    investigation: Annotated[list[AnyMessage], add_messages]  # the search tool conversation
    searches: list[dict]  # each tool search: its source, query and the ids it returned
    assessment: dict
    action: Action
    escalation_reasons: list[str]
    draft: str
    review: dict
    outcome: str
