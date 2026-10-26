"""Graph state and the structured outputs the LLM nodes return.

State holds plain JSON-serialisable values, so model outputs are stored with `.model_dump()`.
"""

from typing import Literal, NotRequired

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Regime = Literal["FOI", "EIR", "SAR", "BAU"]
Decision = Literal["provide", "signpost", "part_refuse", "refuse"]
Action = Literal["send", "escalate", "route"]


class RegimeDecision(BaseModel):
    """Which access regime the request falls under."""

    regime: Regime = Field(
        description=(
            "FOI for a request for recorded information under the Freedom of Information Act; "
            "EIR if the information requested is environmental; SAR if the requester is asking "
            "for their own personal data; BAU for a service request, complaint or routine "
            "enquiry that is not a request for recorded information."
        )
    )
    reason: str = Field(
        description="One or two sentences explaining the choice, citing the guidance relied on."
    )


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


class InputState(TypedDict):
    """What a caller provides to start a case."""

    request: str
    received: NotRequired[str]  # ISO date; defaults to today


class FOIState(InputState, total=False):
    """Full graph state."""

    regime_guidance: list[dict]
    regime: Regime
    regime_reason: str
    deadline: str  # ISO date
    evidence: list[dict]
    assessment: dict
    action: Action
    escalation_reasons: list[str]
    draft: str
    review: dict
    outcome: str
