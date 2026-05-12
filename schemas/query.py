"""Query and search-related schemas for the AI Research Assistant."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class QueryType(str, Enum):
    """Types of queries supported."""
    SIMPLE_QA = "simple_qa"
    RESEARCH = "research"
    SUMMARY = "summary"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    EXTRACTION = "extraction"


class SearchScope(str, Enum):
    """Search scope options."""
    ALL_DOCUMENTS = "all_documents"
    SPECIFIC_DOCUMENTS = "specific_documents"
    BY_CATEGORY = "by_category"
    BY_TAGS = "by_tags"
    BY_DATE_RANGE = "by_date_range"


class SortOrder(str, Enum):
    """Sort order for results."""
    RELEVANCE = "relevance"
    DATE_DESC = "date_desc"
    DATE_ASC = "date_asc"
    TITLE = "title"


class SearchFilter(BaseModel):
    """Filters for document search."""
    
    # Document filters
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs")
    categories: Optional[List[str]] = Field(None, description="Document categories")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    
    # Date filters
    date_from: Optional[datetime] = Field(None, description="Documents from this date")
    date_to: Optional[datetime] = Field(None, description="Documents until this date")
    
    # Content filters
    min_word_count: Optional[int] = Field(None, ge=0, description="Minimum word count")
    max_word_count: Optional[int] = Field(None, ge=0, description="Maximum word count")
    language: Optional[str] = Field(None, description="Document language")
    
    # Search parameters
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results")
    
    @validator('max_word_count')
    def validate_word_count_range(cls, v, values):
        """Ensure max_word_count > min_word_count."""
        min_count = values.get('min_word_count')
        if min_count is not None and v is not None and v <= min_count:
            raise ValueError("max_word_count must be greater than min_word_count")
        return v


class Query(BaseModel):
    """User query with context and parameters."""
    
    query_id: str = Field(..., description="Unique query identifier")
    text: str = Field(..., min_length=1, max_length=1000, description="Query text")
    query_type: QueryType = Field(default=QueryType.SIMPLE_QA, description="Type of query")
    
    # Search configuration
    search_scope: SearchScope = Field(default=SearchScope.ALL_DOCUMENTS, description="Search scope")
    filters: SearchFilter = Field(default_factory=SearchFilter, description="Search filters")
    sort_order: SortOrder = Field(default=SortOrder.RELEVANCE, description="Result sorting")
    
    # Context
    conversation_history: List[str] = Field(default_factory=list, description="Previous messages")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('text')
    def validate_query_text(cls, v):
        """Clean and validate query text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty")
        return cleaned
    
    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        """Limit conversation history length."""
        if len(v) > 20:  # Keep last 20 messages
            return v[-20:]
        return v


class SearchResult(BaseModel):
    """Individual search result."""
    
    document_id: str = Field(..., description="Document identifier")
    chunk_id: Optional[str] = Field(None, description="Specific chunk ID")
    
    # Content
    title: str = Field(..., description="Document title")
    content_snippet: str = Field(..., description="Relevant content snippet")
    
    # Scoring
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    
    # Metadata
    source: str = Field(..., description="Document source")
    document_type: str = Field(..., description="Document type")
    created_at: datetime = Field(..., description="Document creation date")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    
    # Highlighting
    highlighted_text: Optional[str] = Field(None, description="Text with highlighting")
    match_positions: List[Dict[str, int]] = Field(default_factory=list, description="Match positions")


class QueryResult(BaseModel):
    """Complete query result with metadata."""
    
    query_id: str = Field(..., description="Original query ID")
    query_text: str = Field(..., description="Original query text")
    
    # Results
    search_results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    
    # Performance metrics
    search_time_ms: float = Field(..., ge=0, description="Search time in milliseconds")
    embedding_time_ms: Optional[float] = Field(None, ge=0, description="Embedding time")
    
    # Status
    success: bool = Field(default=True, description="Whether query succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Response generation (for when LLM processes results)
    llm_response: Optional[str] = Field(None, description="Generated response")
    sources_used: List[str] = Field(default_factory=list, description="Sources used in response")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    
    @property
    def has_results(self) -> bool:
        """Check if query returned results."""
        return len(self.search_results) > 0
    
    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top search result."""
        return self.search_results[0] if self.search_results else None
    
    def get_results_by_score(self, min_score: float = 0.0) -> List[SearchResult]:
        """Filter results by minimum relevance score."""
        return [r for r in self.search_results if r.relevance_score >= min_score]


# Query builder helpers
def create_simple_query(text: str, **kwargs) -> Query:
    """Create a simple Q&A query."""
    from uuid import uuid4
    
    return Query(
        query_id=str(uuid4()),
        text=text,
        query_type=QueryType.SIMPLE_QA,
        **kwargs
    )


def create_research_query(text: str, max_results: int = 20, **kwargs) -> Query:
    """Create a research query with expanded results."""
    from uuid import uuid4
    
    filters = SearchFilter(max_results=max_results)
    
    return Query(
        query_id=str(uuid4()),
        text=text,
        query_type=QueryType.RESEARCH,
        filters=filters,
        **kwargs
    )


def create_category_query(text: str, categories: List[str], **kwargs) -> Query:
    """Create a query limited to specific categories."""
    from uuid import uuid4
    
    filters = SearchFilter(categories=categories)
    
    return Query(
        query_id=str(uuid4()),
        text=text,
        search_scope=SearchScope.BY_CATEGORY,
        filters=filters,
        **kwargs
    )