"""State definition for the Q&A LangGraph workflow."""

from typing import List, Optional, TypedDict

from schemas.query import Query, SearchResult
from schemas.response import LLMResponse


class QAState(TypedDict):
    """
    Shared state passed between all nodes in the Q&A workflow.

    LangGraph merges the dict returned by each node into this state,
    so every field is Optional — nodes only write what they produce.

    Fields
    ------
    query           : The original Query object (set at graph entry).
    search_results  : Chunks retrieved from ChromaDB.
    context_text    : Formatted string built from search_results for the prompt.
    raw_llm_output  : Plain text returned by the LLM before validation.
    final_response  : Fully validated LLMResponse — the graph's output.
    error           : Soft error message; set by any node that fails gracefully.
    """

    query: Query
    search_results: List[SearchResult]
    context_text: str
    raw_llm_output: str
    final_response: Optional[LLMResponse]
    error: Optional[str]
