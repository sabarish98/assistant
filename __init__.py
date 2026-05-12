"""AI Research Assistant - A comprehensive document processing and semantic search system."""

__version__ = "1.0.0"
__author__ = "AI Research Assistant Team"
__email__ = ""
__description__ = "A comprehensive AI Research Assistant for document processing and semantic search"

# Core imports for easy access
from core.config import config
from core.logger import app_logger
from core.llm_client import ollama_client

# Main components
from ingestion.document_loader import document_loader
from storage.vector_store import vector_store
from storage.embeddings import embedding_manager

# Schema imports
from schemas.document import Document, DocumentMetadata, DocumentChunk
from schemas.query import Query, QueryResult, SearchResult
from schemas.response import LLMResponse, ValidationResult, ConfidenceScore

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Core components
    "config",
    "app_logger", 
    "ollama_client",
    
    # Main services
    "document_loader",
    "vector_store",
    "embedding_manager",
    
    # Schemas
    "Document",
    "DocumentMetadata", 
    "DocumentChunk",
    "Query",
    "QueryResult",
    "SearchResult",
    "LLMResponse",
    "ValidationResult",
    "ConfidenceScore",
]