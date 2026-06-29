"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from .llm import get_llm
from .state import AgentState, make_event, ApprovalDecision


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Pydantic models for LLM Structured Output ───────────────────────
class IntentClassification(BaseModel):
    route: str = Field(
        description="One of: 'simple', 'tool', 'missing_info', 'risky', 'error'."
    )
    risk_level: str = Field(
        description="Set to 'high' for risky routes, 'low' otherwise."
    )


class EvaluationJudge(BaseModel):
    is_success: bool = Field(
        description="True if tool output successfully resolved query, False if it encountered an error or needs retry."
    )
    reasoning: str = Field(description="Brief explanation of evaluation.")


# ─── NODE IMPLEMENTATIONS ──────────────────────────────────────────────

def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM (with offline heuristic fallback)."""
    query = state.get("query", "")
    system_prompt = (
        "You are an expert intent classifier for a support ticket system.\n"
        "Classify the user query into exactly one of these routes:\n"
        "- 'risky': Actions with side effects (refunds, deleting accounts/data, sending emails, cancellations).\n"
        "- 'tool': Information lookups (order status, tracking, database queries).\n"
        "- 'missing_info': Vague or incomplete queries lacking actionable context (e.g., 'Can you fix it?', 'Help me').\n"
        "- 'error': System failures, timeouts, crashes, or service unavailable reports.\n"
        "- 'simple': General questions answerable without tools or actions (e.g., 'How do I reset my password?').\n\n"
        "Priority guide: risky > tool > missing_info > error > simple.\n"
        "Set risk_level to 'high' if route is 'risky', otherwise 'low'."
    )
    
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(IntentClassification)
        res: IntentClassification = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {query}")
        ])
        route = res.route.lower()
        if route not in ["simple", "tool", "missing_info", "risky", "error"]:
            route = "simple"
        risk_level = "high" if route == "risky" else res.risk_level.lower()
    except Exception:
        # Fallback heuristic for offline mode or when no LLM API key is provided
        q = query.lower()
        if any(k in q for k in ["refund", "delete", "cancel", "email"]):
            route = "risky"
            risk_level = "high"
        elif any(k in q for k in ["lookup", "order", "status", "search", "track"]):
            route = "tool"
            risk_level = "low"
        elif any(k in q for k in ["timeout", "failure", "error", "crash", "system failure"]):
            route = "error"
            risk_level = "low"
        elif "fix it" in q or "help me" in q or len(q.split()) <= 3:
            route = "missing_info"
            risk_level = "low"
        else:
            route = "simple"
            risk_level = "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"classified as {route} (risk: {risk_level})")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result_string = f"ERROR: Timeout failure while processing tool request for query '{query}' (attempt {attempt})"
    else:
        result_string = f"Tool Execution Success: Processed request for '{query}' successfully."

    return {
        "tool_results": [result_string],
        "events": [make_event("tool", "completed", f"tool result generated: {result_string[:30]}")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate (using LLM-as-judge with fallback)."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""
    
    eval_result = "success"
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(EvaluationJudge)
        res: EvaluationJudge = structured_llm.invoke([
            SystemMessage(content="Evaluate whether the tool result represents a successful execution or an error/failure requiring a retry."),
            HumanMessage(content=f"Tool Output: {latest_result}")
        ])
        eval_result = "success" if res.is_success else "needs_retry"
    except Exception:
        # Fallback to heuristic check
        if "ERROR" in latest_result:
            eval_result = "needs_retry"
        else:
            eval_result = "success"

    return {
        "evaluation_result": eval_result,
        "events": [make_event("evaluate", "completed", f"evaluated as {eval_result}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM (with offline fallback)."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval", {})

    context_parts = [f"User Query: {query}"]
    if tool_results:
        context_parts.append(f"Tool Results: {' | '.join(tool_results)}")
    if approval:
        context_parts.append(f"Approval Decision: {approval}")

    context = "\n".join(context_parts)
    system_prompt = "You are a helpful customer support AI agent. Provide a clear, polite, and grounded response based strictly on the provided context."

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}")
        ])
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        answer_text = f"Processed support request for query '{query}'. Action completed successfully based on available tools and verification."

    return {
        "final_answer": answer_text,
        "events": [make_event("answer", "completed", "generated grounded response")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    try:
        llm = get_llm(temperature=0.2)
        res = llm.invoke([
            SystemMessage(content="The user's query is vague or missing details. Politely ask a specific clarification question to get the required information."),
            HumanMessage(content=f"Vague Query: {query}")
        ])
        question = res.content if hasattr(res, "content") else str(res)
    except Exception:
        question = f"Could you please provide more details regarding your request: '{query}'?"

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "requested clarification")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = f"Proposed high-risk action requiring verification for query: '{query}'."
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "completed", "prepared risky action for approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step."""
    use_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true"
    
    if use_interrupt:
        try:
            from langgraph.types import interrupt
            decision_data = interrupt({
                "proposed_action": state.get("proposed_action"),
                "query": state.get("query")
            })
            if isinstance(decision_data, dict):
                decision = decision_data
            else:
                decision = {"approved": bool(decision_data), "reviewer": "hitl-user", "comment": "HITL interrupt response"}
        except ImportError:
            decision = ApprovalDecision(approved=True, reviewer="mock-reviewer", comment="Mock approved (interrupt unavailable)").model_dump()
    else:
        decision = ApprovalDecision(approved=True, reviewer="mock-reviewer", comment="Mock approved automatically").model_dump()

    return {
        "approval": decision,
        "events": [make_event("approval", "completed", f"approval decision recorded: {decision}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt."""
    attempt = state.get("attempt", 0) + 1
    error_msg = f"Attempt {attempt} encountered failure; retrying operation."
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "completed", f"retry attempt incremented to {attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    query = state.get("query", "")
    answer = f"System error: Unable to complete request '{query}' after maximum retry attempts were exceeded."
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "completed", "routed to dead letter queue")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
