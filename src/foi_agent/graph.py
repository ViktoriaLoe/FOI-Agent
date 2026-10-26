"""The FOI triage graph. See README.md for the flow."""

from langchain_typesafe.client import TypeSafeAPIConnectionError, TypeSafeInternalServerError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from foi_agent import nodes, rules
from foi_agent.prompts import PROMPT_VERSION
from foi_agent.retrieval import corpus_version
from foi_agent.state import FOIState, InputState


def run_metadata() -> dict:
    """What produced a run. Attached to every trace, so runs and experiments can be filtered
    and compared by model, prompt version, retriever and knowledge base."""
    return {
        "model": nodes.model_name(),
        "classifier_model": nodes.classifier_model_name(),
        "prompt_version": PROMPT_VERSION,
        "retriever": "bm25",
        "retrieval_k": nodes.RETRIEVAL_K,
        "corpus_version": corpus_version(),
    }


def build_graph(checkpointer=None):
    """Compile the graph.

    Nodes are looked up through their modules at build time, so tests can monkeypatch them.
    """
    llm_retry = RetryPolicy(max_attempts=3)
    # The default policy retries every TypeSafe error, so a bad key, an unknown model or a rate
    # limit would be sent three times. Retry only server errors, dropped connections and timeouts.
    classifier_retry = RetryPolicy(
        max_attempts=3, retry_on=(TypeSafeInternalServerError, TypeSafeAPIConnectionError)
    )
    builder = StateGraph(FOIState, input_schema=InputState)

    builder.add_node("retrieve_regime_guidance", nodes.retrieve_regime_guidance)
    builder.add_node("classify_regime", nodes.classify_regime, retry_policy=classifier_retry)
    builder.add_node("route_non_foi", rules.route_non_foi)
    builder.add_node("compute_deadline", nodes.compute_deadline)
    builder.add_node("retrieve_evidence", nodes.retrieve_evidence)
    builder.add_node("assess", nodes.assess, retry_policy=llm_retry)
    builder.add_node("decide_action", rules.decide_action)
    builder.add_node("draft_reply", nodes.draft_reply, retry_policy=llm_retry)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("log_and_send", nodes.log_and_send)

    builder.add_edge(START, "retrieve_regime_guidance")
    builder.add_edge("retrieve_regime_guidance", "classify_regime")
    builder.add_conditional_edges(
        "classify_regime", rules.route_by_regime, ["compute_deadline", "route_non_foi"]
    )
    builder.add_conditional_edges(
        "route_non_foi", rules.route_by_action, ["human_review", "log_and_send"]
    )
    builder.add_edge("compute_deadline", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "assess")
    builder.add_edge("assess", "decide_action")
    builder.add_edge("decide_action", "draft_reply")
    builder.add_conditional_edges(
        "draft_reply", rules.route_by_action, ["human_review", "log_and_send"]
    )
    # human_review chooses its next step with Command, so it has no static edge out.
    builder.add_edge("log_and_send", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph.with_config({"run_name": "foi_triage", "metadata": run_metadata()})


# For LangGraph Studio and `langgraph dev`, which supply their own checkpointer.
graph = build_graph()
