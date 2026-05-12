"""Schema definitions for AI Research Assistant."""

from .document import Document, DocumentMetadata, DocumentChunk
from .query import Query, QueryResult, SearchFilter
from .response import LLMResponse, ValidationResult, ConfidenceScore

__all__ = [
    "Document", "DocumentMetadata", "DocumentChunk",
    "Query", "QueryResult", "SearchFilter", 
    "LLMResponse", "ValidationResult", "ConfidenceScore"
]