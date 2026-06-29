# Day 08 Lab Report

## 1. Team / student

- Name: Antigravity AI & Student
- Repo/commit: phase2-track3-day8-langgraph-agent
- Date: 2026-06-29

## 2. Architecture

The support-ticket agent is built using LangGraph's `StateGraph` architecture. The workflow comprises 11 nodes:
1. `intake`: Normalizes incoming user queries.
2. `classify`: Uses LLM structured output to classify intent into one of 5 routes (`simple`, `tool`, `missing_info`, `risky`, `error`).
3. `tool`: Executes mock tool calls and simulates transient errors.
4. `evaluate`: Evaluates tool results to determine if retries are needed (retry-loop gate).
5. `answer`: Uses LLM grounded generation to answer queries.
6. `clarify`: Requests clarification for vague or incomplete queries.
7. `risky_action`: Prepares sensitive operations (refunds, deletions) for human review.
8. `approval`: Human-in-the-loop (HITL) checkpoint supporting mock approval and real interrupts (`LANGGRAPH_INTERRUPT`).
9. `retry`: Handles bounded retry attempts with attempt counting.
10. `dead_letter`: Escalates unresolvable failures after retry exhaustion.
11. `finalize`: Emits standardized audit logs prior to completion (`END`).

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append (`operator.add`) | Audit log of system messages |
| tool_results | append (`operator.add`) | Record of tool execution outputs |
| errors | append (`operator.add`) | List of captured error traces |
| events | append (`operator.add`) | Structured timeline events |
| route | overwrite | Current graph routing intent |
| attempt | overwrite | Current retry counter |
| final_answer | overwrite | Grounded final output for user |
| evaluation_result| overwrite | Gate decision for retry loop |

## 4. Scenario results

**Summary Metrics:**
- Total Scenarios: 7
- Success Rate: 100.00%
- Avg Nodes Visited: 6.43
- Total Retries: 3
- Total Interrupts: 2

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Pass | 0 | 0 |
| S02_tool | tool | tool | Pass | 0 | 0 |
| S03_missing | missing_info | missing_info | Pass | 0 | 0 |
| S04_risky | risky | risky | Pass | 0 | 1 |
| S05_error | error | error | Pass | 2 | 0 |
| S06_delete | risky | risky | Pass | 0 | 1 |
| S07_dead_letter | error | error | Pass | 1 | 0 |

## 5. Failure analysis

1. **Retry or tool failure**: Transient errors in external tool calls are caught by `evaluate_node`. If an error occurs, the state increments `attempt` via `retry_node`. If `attempt < max_attempts`, execution retries `tool_node`. Once `attempt >= max_attempts`, execution safely exits to `dead_letter_node` to prevent infinite loops.
2. **Risky action without approval**: Actions with side effects (e.g., refunds, data deletion) are classified as `risky`. The workflow routes through `risky_action_node` to `approval_node`. If approval is rejected, the graph routes to `clarify_node` rather than executing the destructive tool action.

## 6. Persistence / recovery evidence

The system integrates `SqliteSaver` and `MemorySaver` checkpointers. Each run receives a unique `thread_id`. Checkpointing preserves exact graph state at every step, enabling state inspection and crash recovery.

## 7. Extension work

- **SQLite Checkpointer**: Implemented persistent checkpointing using SQLite in WAL mode.
- **LLM Structured Output & LLM-as-Judge**: Integrated structured schema outputs for intent classification and evaluation.
- **HITL Integration**: Configured `interrupt()` handling for production approval workflows.

## 8. Improvement plan

In a production deployment, we would:
1. Replace mock tool implementations with actual API integrations (vector search, SQL database).
2. Implement parallel fan-out tool execution using LangGraph `Send()`.
3. Add full open telemetry tracing (LangSmith) for fine-grained monitoring.
