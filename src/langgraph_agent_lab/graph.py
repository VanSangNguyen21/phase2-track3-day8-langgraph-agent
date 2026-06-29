"""Graph construction."""

from __future__ import annotations

from typing import Any
from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow."""
    from langgraph.graph import StateGraph, START, END
    from .nodes import (
        intake_node,
        classify_node,
        tool_node,
        evaluate_node,
        answer_node,
        ask_clarification_node,
        risky_action_node,
        approval_node,
        retry_or_fallback_node,
        dead_letter_node,
        finalize_node,
    )
    from .routing import (
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
        route_after_approval,
    )

    workflow = StateGraph(AgentState)

    # 1. Register nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("clarify", ask_clarification_node)
    workflow.add_node("risky_action", risky_action_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("retry", retry_or_fallback_node)
    workflow.add_node("dead_letter", dead_letter_node)
    workflow.add_node("finalize", finalize_node)

    # 2. Wire fixed edges from START
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classify")

    # 3. Conditional routing after classify
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "tool": "tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )

    # 4. Tool execution & evaluation loop
    workflow.add_edge("tool", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "answer": "answer",
            "retry": "retry",
        },
    )

    # 5. Retry loop routing
    workflow.add_conditional_edges(
        "retry",
        route_after_retry,
        {
            "tool": "tool",
            "dead_letter": "dead_letter",
        },
    )

    # 6. Risky action & approval flow
    workflow.add_edge("risky_action", "approval")
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "tool": "tool",
            "clarify": "clarify",
        },
    )

    # 7. Finalize and END edges
    workflow.add_edge("answer", "finalize")
    workflow.add_edge("clarify", "finalize")
    workflow.add_edge("dead_letter", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)
