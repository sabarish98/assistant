"""
LangGraph Q&A Workflow for AI Research Assistant.

Graph shape
-----------

    START
      │
      ▼
  query_analysis          ← classify query, tune retrieval params
      │
      ▼
  document_retrieval      ← semantic search in ChromaDB
      │
      ├─ results found ──► generate_response   ← RAG prompt → Ollama
      │                         │
      │                         ▼
      │                   validate_and_format  ← build LLMResponse
      │                         │
      └─ no results ──► handle_empty           ← graceful fallback
                               │
                              END (both paths)
"""

import time
import uuid
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core.config import config
from core.llm_client import ollama_client
from core.logger import app_logger
from prompts.registry import prompt_registry
from schemas.query import QueryType, SearchResult
from schemas.response import (
    ConfidenceLevel,
    ConfidenceScore,
    LLMResponse,
    ResponseType,
    SourceCitation,
    ValidationResult,
    ValidationStatus,
    create_error_response,
    create_successful_response,
)
from storage.vector_store import vector_store
from workflows.state import QAState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# How many chunks to retrieve per query type
_RETRIEVAL_LIMITS: dict[QueryType, int] = {
    QueryType.SIMPLE_QA: 5,
    QueryType.RESEARCH: 15,
    QueryType.SUMMARY: 10,
    QueryType.SYNTHESIS: 12,
    QueryType.COMPARISON: 10,
    QueryType.EXTRACTION: 8,
}

# Resolved once at import time — pin a version with "qa_system@v1" or use
# "qa_system" to always pick up the latest version automatically.
_QA_SYSTEM_PROMPT_REF = "qa_system"


def _format_context(results: list[SearchResult]) -> str:
    """Turn a list of SearchResult objects into a numbered context block."""
    if not results:
        return ""

    lines = ["### Retrieved Context\n"]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] Source: {r.title} (relevance: {r.similarity_score:.2f})\n"
            f"{r.content_snippet}\n"
        )
    return "\n".join(lines)


def _compute_confidence(
    results: list[SearchResult],
    raw_output: str,
) -> ConfidenceScore:
    """
    Derive a confidence score from retrieval quality and response length.

    Heuristics (simple but transparent):
    - source_reliability  : mean similarity score of top results
    - content_relevance   : penalises very short responses (< 50 chars)
    - overall_score       : weighted average of the two
    """
    if not results:
        return ConfidenceScore(
            overall_score=0.0,
            level=ConfidenceLevel.LOW,
            source_reliability=0.0,
            content_relevance=0.0,
            uncertainty_areas=["No source documents found"],
        )

    source_reliability = sum(r.similarity_score for r in results) / len(results)

    # Penalise very short responses — likely a refusal or incomplete answer
    content_relevance = min(1.0, len(raw_output.strip()) / 200)

    overall = 0.6 * source_reliability + 0.4 * content_relevance
    overall = round(min(1.0, max(0.0, overall)), 3)

    level = (
        ConfidenceLevel.HIGH
        if overall >= 0.8
        else ConfidenceLevel.MEDIUM
        if overall >= 0.5
        else ConfidenceLevel.LOW
    )

    factors = [f"Retrieved {len(results)} source chunk(s)"]
    uncertainties = []

    if source_reliability < 0.6:
        uncertainties.append("Low average similarity score — sources may not be closely related")
    if content_relevance < 0.5:
        uncertainties.append("Response is short — may be incomplete")

    return ConfidenceScore(
        overall_score=overall,
        level=level,
        source_reliability=round(source_reliability, 3),
        content_relevance=round(content_relevance, 3),
        confidence_factors=factors,
        uncertainty_areas=uncertainties,
    )


def _build_source_citations(results: list[SearchResult]) -> list[SourceCitation]:
    """Convert SearchResult objects to SourceCitation objects."""
    return [
        SourceCitation(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            title=r.title,
            source=r.source,
            quoted_text=r.content_snippet,
            relevance_score=r.similarity_score,
            document_type=r.document_type,
            created_at=r.created_at,
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# Node 1 — query_analysis
# ---------------------------------------------------------------------------

def query_analysis(state: QAState) -> QAState:
    """
    Inspect the query and tune retrieval parameters.

    Currently adjusts max_results based on query type so that research
    queries cast a wider net than simple Q&A.
    """
    query = state["query"]
    app_logger.info(f"[query_analysis] type={query.query_type}, text='{query.text[:60]}'")

    # Tune max_results for the query type
    desired_limit = _RETRIEVAL_LIMITS.get(query.query_type, 10)
    query.filters.max_results = desired_limit

    app_logger.debug(f"[query_analysis] max_results set to {desired_limit}")
    return {"query": query}


# ---------------------------------------------------------------------------
# Node 2 — document_retrieval
# ---------------------------------------------------------------------------

def document_retrieval(state: QAState) -> QAState:
    """
    Run semantic search against ChromaDB and format results as context text.
    """
    query = state["query"]
    app_logger.info(f"[document_retrieval] searching for '{query.text[:60]}'")

    results = vector_store.search_similar(
        query_text=query.text,
        n_results=query.filters.max_results,
    )

    # Filter by similarity threshold
    threshold = query.filters.similarity_threshold
    filtered = [r for r in results if r.similarity_score >= threshold]

    app_logger.info(
        f"[document_retrieval] {len(results)} raw results, "
        f"{len(filtered)} above threshold {threshold}"
    )

    context_text = _format_context(filtered)

    return {
        "search_results": filtered,
        "context_text": context_text,
    }


# ---------------------------------------------------------------------------
# Node 3 — generate_response
# ---------------------------------------------------------------------------

def generate_response(state: QAState) -> QAState:
    """
    Build a RAG prompt and call Ollama synchronously.

    Uses the formatted context from document_retrieval and the original
    query text to construct a system + human message pair.

    The system prompt is loaded from the PromptRegistry so it can be
    versioned and iterated without touching this file.
    """
    query = state["query"]
    context_text = state.get("context_text", "")

    app_logger.info(f"[generate_response] invoking {config.ollama_model}")

    # Load system prompt from registry (latest version unless pinned)
    system_prompt_text = prompt_registry.get(_QA_SYSTEM_PROMPT_REF).template

    prompt = (
        f"{context_text}\n\n"
        f"### Question\n{query.text}\n\n"
        f"### Answer"
    )

    messages = [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=prompt),
    ]

    try:
        t0 = time.time()
        response = ollama_client.invoke(messages)
        elapsed_ms = (time.time() - t0) * 1000

        raw_output = response.content.strip()
        app_logger.info(
            f"[generate_response] received {len(raw_output)} chars in {elapsed_ms:.0f}ms"
        )
        return {"raw_llm_output": raw_output}

    except Exception as exc:
        app_logger.error(f"[generate_response] LLM call failed: {exc}")
        return {
            "raw_llm_output": "",
            "error": f"LLM generation failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Node 4 — validate_and_format
# ---------------------------------------------------------------------------

def validate_and_format(state: QAState) -> QAState:
    """
    Wrap the raw LLM output in a fully typed LLMResponse.

    Scores confidence from retrieval quality, builds SourceCitation list,
    and flags any soft errors captured earlier in the graph.
    """
    query = state["query"]
    raw_output = state.get("raw_llm_output", "")
    results = state.get("search_results", [])
    soft_error = state.get("error")

    app_logger.info("[validate_and_format] building final response")

    # --- Validation checks ---
    errors: list[str] = []
    warnings: list[str] = []

    if soft_error:
        errors.append(soft_error)

    if not raw_output.strip():
        errors.append("LLM returned an empty response")

    if len(raw_output) < 20:
        warnings.append("Response is unusually short")

    is_valid = len(errors) == 0
    validation = ValidationResult(
        status=ValidationStatus.VALID if is_valid else ValidationStatus.INVALID,
        is_valid=is_valid,
        schema_validation=True,
        content_validation=bool(raw_output.strip()),
        safety_validation=True,   # placeholder — Stage 3 adds real safety checks
        errors=errors,
        warnings=warnings,
    )

    # --- Confidence scoring ---
    confidence = _compute_confidence(results, raw_output)

    # --- Source citations ---
    sources = _build_source_citations(results)

    # --- Assemble LLMResponse ---
    content = raw_output if raw_output.strip() else (
        "I was unable to generate a response. Please try rephrasing your question."
    )

    final_response = LLMResponse(
        response_id=str(uuid.uuid4()),
        query_id=query.query_id,
        content=content,
        response_type=ResponseType.ANSWER,
        confidence=confidence,
        validation=validation,
        sources=sources,
        source_count=len(sources),
        model_name=config.ollama_model,
        temperature=config.ollama_temperature,
        generation_time_ms=0.0,   # timing tracked per-node; can be wired in later
        created_at=datetime.now(),
    )

    app_logger.info(
        f"[validate_and_format] confidence={confidence.overall_score}, "
        f"valid={is_valid}, sources={len(sources)}"
    )

    return {"final_response": final_response}


# ---------------------------------------------------------------------------
# Node 5 — handle_empty
# ---------------------------------------------------------------------------

def handle_empty(state: QAState) -> QAState:
    """
    Return a graceful response when no documents were retrieved.

    This is a terminal node — it writes final_response and the graph ends.
    """
    query = state["query"]
    app_logger.warning(f"[handle_empty] no results for '{query.text[:60]}'")

    final_response = create_error_response(
        query_id=query.query_id,
        error_message=(
            "No relevant documents were found in your knowledge base for this query. "
            "Try ingesting more documents or rephrasing your question."
        ),
        model_name=config.ollama_model,
        generation_time_ms=0.0,
    )

    return {"final_response": final_response}


# ---------------------------------------------------------------------------
# Conditional edge — route after document_retrieval
# ---------------------------------------------------------------------------

def route_after_retrieval(
    state: QAState,
) -> Literal["generate_response", "handle_empty"]:
    """
    Route to generate_response if we have results, otherwise handle_empty.
    """
    results = state.get("search_results", [])
    if results:
        app_logger.debug("[route] results found → generate_response")
        return "generate_response"
    app_logger.debug("[route] no results → handle_empty")
    return "handle_empty"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_qa_graph() -> StateGraph:
    """Assemble and compile the Q&A StateGraph."""

    graph = StateGraph(QAState)

    # Register nodes
    graph.add_node("query_analysis", query_analysis)
    graph.add_node("document_retrieval", document_retrieval)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_and_format", validate_and_format)
    graph.add_node("handle_empty", handle_empty)

    # Edges
    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "document_retrieval")

    # Conditional branch after retrieval
    graph.add_conditional_edges(
        "document_retrieval",
        route_after_retrieval,
        {
            "generate_response": "generate_response",
            "handle_empty": "handle_empty",
        },
    )

    graph.add_edge("generate_response", "validate_and_format")
    graph.add_edge("validate_and_format", END)
    graph.add_edge("handle_empty", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Compiled graph — import this in the CLI and tests
qa_graph = build_qa_graph()


def run_qa(query_text: str, **query_kwargs) -> LLMResponse:
    """
    Convenience wrapper: run the Q&A graph for a plain text question.

    Parameters
    ----------
    query_text   : The question to answer.
    **query_kwargs : Passed to create_simple_query (e.g. tags, category filters).

    Returns
    -------
    LLMResponse with content, confidence, sources, and validation fields.
    """
    from schemas.query import create_simple_query

    query = create_simple_query(query_text, **query_kwargs)

    app_logger.info(f"[run_qa] starting graph for query_id={query.query_id}")

    initial_state: QAState = {
        "query": query,
        "search_results": [],
        "context_text": "",
        "raw_llm_output": "",
        "final_response": None,
        "error": None,
    }

    final_state = qa_graph.invoke(initial_state)

    response = final_state.get("final_response")
    if response is None:
        # Should never happen, but be defensive
        response = create_error_response(
            query_id=query.query_id,
            error_message="Graph completed without producing a response.",
            model_name=config.ollama_model,
            generation_time_ms=0.0,
        )

    return response


__all__ = ["qa_graph", "run_qa", "build_qa_graph", "QAState"]
