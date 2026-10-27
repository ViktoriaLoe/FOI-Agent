"""The FOI triage graph. See README.md for the flow."""

import os

from langchain_typesafe.client import TypeSafeAPIConnectionError, TypeSafeInternalServerError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from foi_agent import nodes, rules
from foi_agent.prompts import PROMPT_VERSION
from foi_agent.retrieval import corpus_version
from foi_agent.state import FOIState, InputState


def search_tool_enabled() -> bool:
    """Whether the graph includes the search tool. On unless FOI_SEARCH_TOOL is 0, false or off."""
    return os.environ.get("FOI_SEARCH_TOOL", "").strip().lower() not in {"0", "false", "off"}


def run_metadata(search_tool: bool | None = None) -> dict:
    """What produced a run. Attached to every trace, so runs and experiments can be filtered
    and compared by model, prompt version, retriever, search tool and knowledge base.

    `search_tool` is read from FOI_SEARCH_TOOL when None, as in build_graph.
    """
    if search_tool is None:
        search_tool = search_tool_enabled()
    return {
        "model": nodes.model_name(),
        "classifier_model": nodes.classifier_model_name(),
        "prompt_version": PROMPT_VERSION,
        "retriever": "bm25",
        "retrieval_k": nodes.RETRIEVAL_K,
        "corpus_version": corpus_version(),
        "search_tool": search_tool,
        "max_searches": rules.MAX_SEARCHES,
    }


def build_graph(checkpointer=None, search_tool: bool | None = None):
    """Compile the graph.

    With the search tool, evidence retrieval is followed by up to MAX_SEARCHES searches the model
    asks for. Without it, as in v1, retrieval goes straight to assess. `search_tool` is read from
    FOI_SEARCH_TOOL when None.

    Nodes are looked up through their modules at build time, so tests can monkeypatch them.
    """
    if search_tool is None:
        search_tool = search_tool_enabled()
    llm_retry = RetryPolicy(max_attempts=3)
    # The default policy retries every TypeSafe error, so a bad key, an unknown model or a rate
    # limit would be sent three times. Retry only server errors, dropped connections and timeouts.
    classifier_retry = RetryPolicy(
        max_attempts=3, retry_on=(TypeSafeInternalServerError, TypeSafeAPIConnectionError)
    )
    builder = StateGraph(FOIState, input_schema=InputState)

    builder.add_node("classify_regime", nodes.classify_regime, retry_policy=classifier_retry)
    builder.add_node("check_regime", nodes.check_regime, retry_policy=llm_retry)
    builder.add_node("compute_deadline", nodes.compute_deadline)
    builder.add_node("route_non_foi", rules.route_non_foi)
    builder.add_node("retrieve_evidence", nodes.retrieve_evidence)
    if search_tool:
        builder.add_node("investigate", nodes.investigate, retry_policy=llm_retry)
        builder.add_node("run_searches", nodes.run_searches)
    builder.add_node("assess", nodes.assess, retry_policy=llm_retry)
    builder.add_node("decide_action", rules.decide_action)
    builder.add_node("draft_reply", nodes.draft_reply, retry_policy=llm_retry)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("log_and_send", nodes.log_and_send)

    # Every case gets the second opinion and its deadline before routing, so a case escalated
    # before FOI work still reaches the officer with both.
    builder.add_edge(START, "classify_regime")
    builder.add_edge("classify_regime", "check_regime")
    builder.add_edge("check_regime", "compute_deadline")
    builder.add_conditional_edges(
        "compute_deadline", rules.route_by_regime, ["retrieve_evidence", "route_non_foi"]
    )
    builder.add_conditional_edges(
        "route_non_foi", rules.route_by_action, ["human_review", "log_and_send"]
    )
    if search_tool:
        builder.add_edge("retrieve_evidence", "investigate")
        builder.add_conditional_edges(
            "investigate", rules.route_after_investigate, ["run_searches", "assess"]
        )
        builder.add_edge("run_searches", "investigate")
    else:
        builder.add_edge("retrieve_evidence", "assess")
    builder.add_edge("assess", "decide_action")
    builder.add_edge("decide_action", "draft_reply")
    builder.add_conditional_edges(
        "draft_reply", rules.route_by_action, ["human_review", "log_and_send"]
    )
    # human_review chooses its next step with Command, so it has no static edge out.
    builder.add_edge("log_and_send", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph.with_config({"run_name": "foi_triage", "metadata": run_metadata(search_tool)})


# For LangGraph Studio and `langgraph dev`, which supply their own checkpointer.
graph = build_graph()
