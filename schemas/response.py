"""Response and validation schemas for the AI Research Assistant."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class ResponseType(str, Enum):
    """Types of LLM responses."""
    ANSWER = "answer"
    SUMMARY = "summary"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    ERROR = "error"
    CLARIFICATION = "clarification"


class ValidationStatus(str, Enum):
    """Response validation status."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"


class ConfidenceLevel(str, Enum):
    """Confidence levels for responses."""
    HIGH = "high"       # 0.8 - 1.0
    MEDIUM = "medium"   # 0.5 - 0.79
    LOW = "low"         # 0.0 - 0.49


class ConfidenceScore(BaseModel):
    """Detailed confidence scoring."""
    
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    level: ConfidenceLevel = Field(..., description="Confidence level category")
    
    # Component scores
    source_reliability: float = Field(..., ge=0.0, le=1.0, description="Source reliability score")
    content_relevance: float = Field(..., ge=0.0, le=1.0, description="Content relevance score")
    factual_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Factual accuracy score")
    
    # Reasoning
    confidence_factors: List[str] = Field(default_factory=list, description="Factors affecting confidence")
    uncertainty_areas: List[str] = Field(default_factory=list, description="Areas of uncertainty")
    
    @validator('level')
    def determine_level(cls, v, values):
        """Automatically determine confidence level from score."""
        score = values.get('overall_score', 0.0)
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class ValidationResult(BaseModel):
    """Result of response validation."""
    
    status: ValidationStatus = Field(..., description="Validation status")
    is_valid: bool = Field(..., description="Whether response passed validation")
    
    # Validation details
    schema_validation: bool = Field(default=True, description="Schema validation passed")
    content_validation: bool = Field(default=True, description="Content validation passed")
    safety_validation: bool = Field(default=True, description="Safety validation passed")
    
    # Issues found
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    # Validation metadata
    validator_version: str = Field(default="1.0.0", description="Validator version")
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    
    @property
    def has_errors(self) -> bool:
        """Check if validation found errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation found warnings."""
        return len(self.warnings) > 0


class SourceCitation(BaseModel):
    """Source citation with metadata."""
    
    document_id: str = Field(..., description="Source document ID")
    chunk_id: Optional[str] = Field(None, description="Specific chunk ID")
    
    # Citation details
    title: str = Field(..., description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    source: str = Field(..., description="Document source")
    
    # Content reference
    quoted_text: Optional[str] = Field(None, description="Exact quoted text")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to response")
    
    # Metadata
    document_type: str = Field(..., description="Type of source document")
    created_at: datetime = Field(..., description="Document creation date")


class LLMResponse(BaseModel):
    """Complete LLM response with metadata and validation."""
    
    response_id: str = Field(..., description="Unique response identifier")
    query_id: str = Field(..., description="Related query ID")
    
    # Core response
    content: str = Field(..., min_length=1, description="Response content")
    response_type: ResponseType = Field(default=ResponseType.ANSWER, description="Type of response")
    
    # Confidence and validation
    confidence: ConfidenceScore = Field(..., description="Confidence scoring")
    validation: ValidationResult = Field(..., description="Validation results")
    
    # Sources and citations
    sources: List[SourceCitation] = Field(default_factory=list, description="Source citations")
    source_count: int = Field(default=0, ge=0, description="Number of sources used")
    
    # Generation metadata
    model_name: str = Field(..., description="Model used for generation")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Generation temperature")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens used in generation")
    generation_time_ms: float = Field(..., ge=0, description="Generation time in milliseconds")
    
    # Response flags
    requires_clarification: bool = Field(default=False, description="Whether clarification is needed")
    contains_speculation: bool = Field(default=False, description="Whether response contains speculation")
    partial_answer: bool = Field(default=False, description="Whether this is a partial answer")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Response creation time")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate response content."""
        if not v.strip():
            raise ValueError("Response content cannot be empty")
        return v.strip()
    
    @validator('source_count')
    def sync_source_count(cls, v, values):
        """Sync source count with actual sources."""
        sources = values.get('sources', [])
        return len(sources)
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if response has high confidence."""
        return self.confidence.level == ConfidenceLevel.HIGH
    
    @property
    def is_valid_response(self) -> bool:
        """Check if response passed all validations."""
        return self.validation.is_valid and not self.validation.has_errors
    
    @property
    def has_reliable_sources(self) -> bool:
        """Check if response has reliable source backing."""
        return len(self.sources) > 0 and self.confidence.source_reliability >= 0.7
    
    def get_primary_sources(self, min_relevance: float = 0.8) -> List[SourceCitation]:
        """Get primary sources above relevance threshold."""
        return [s for s in self.sources if s.relevance_score >= min_relevance]
    
    def add_source(self, source: SourceCitation) -> None:
        """Add a source citation."""
        self.sources.append(source)
        self.source_count = len(self.sources)


# Response builder helpers
def create_successful_response(
    query_id: str,
    content: str,
    confidence_score: float,
    model_name: str,
    **kwargs
) -> LLMResponse:
    """Create a successful response with basic confidence."""
    from uuid import uuid4
    
    confidence = ConfidenceScore(
        overall_score=confidence_score,
        level=ConfidenceLevel.HIGH if confidence_score >= 0.8 else 
              ConfidenceLevel.MEDIUM if confidence_score >= 0.5 else 
              ConfidenceLevel.LOW,
        source_reliability=kwargs.get('source_reliability', confidence_score),
        content_relevance=kwargs.get('content_relevance', confidence_score)
    )
    
    validation = ValidationResult(
        status=ValidationStatus.VALID,
        is_valid=True
    )
    
    # Separate LLMResponse-specific kwargs
    response_fields = LLMResponse.__fields__.keys()
    response_kwargs = {k: v for k, v in kwargs.items() if k in response_fields}
    
    # Build response data
    response_data = {
        'response_id': str(uuid4()),
        'query_id': query_id,
        'content': content,
        'confidence': confidence,
        'validation': validation,
        'model_name': model_name,
        'temperature': kwargs.get('temperature', 0.1),
        'generation_time_ms': kwargs.get('generation_time_ms', 0.0),
        **response_kwargs
    }
    
    return LLMResponse(**response_data)


def create_error_response(
    query_id: str,
    error_message: str,
    model_name: str,
    **kwargs
) -> LLMResponse:
    """Create an error response."""
    from uuid import uuid4
    
    confidence = ConfidenceScore(
        overall_score=0.0,
        level=ConfidenceLevel.LOW,
        source_reliability=0.0,
        content_relevance=0.0,
        uncertainty_areas=["Error in processing"]
    )
    
    validation = ValidationResult(
        status=ValidationStatus.INVALID,
        is_valid=False,
        errors=[error_message]
    )
    
    # Separate LLMResponse-specific kwargs
    response_fields = LLMResponse.__fields__.keys()
    response_kwargs = {k: v for k, v in kwargs.items() if k in response_fields}
    
    # Build response data
    response_data = {
        'response_id': str(uuid4()),
        'query_id': query_id,
        'content': f"I encountered an error while processing your request: {error_message}",
        'response_type': ResponseType.ERROR,
        'confidence': confidence,
        'validation': validation,
        'model_name': model_name,
        'temperature': kwargs.get('temperature', 0.1),
        'generation_time_ms': kwargs.get('generation_time_ms', 0.0),
        **response_kwargs
    }
    
    return LLMResponse(**response_data)