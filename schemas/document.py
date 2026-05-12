"""Document-related schemas for the AI Research Assistant."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    WEB_ARTICLE = "web_article"
    NOTE = "note"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    """Metadata for documents with validation."""
    
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    author: Optional[str] = Field(None, max_length=200, description="Document author")
    source: str = Field(..., description="Document source (file path, URL, etc.)")
    document_type: DocumentType = Field(..., description="Type of document")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(None, description="Last modification timestamp")
    ingested_at: datetime = Field(default_factory=datetime.now, description="Ingestion timestamp")
    
    # Content metadata
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    language: str = Field(default="en", description="Document language")
    
    # Tags and categories
    tags: List[str] = Field(default_factory=list, description="Document tags")
    category: Optional[str] = Field(None, description="Document category")
    
    # Processing metadata
    chunk_count: Optional[int] = Field(None, ge=0, description="Number of chunks created")
    embedding_model: Optional[str] = Field(None, description="Model used for embeddings")
    
    @validator('source')
    def validate_source(cls, v):
        """Validate source format."""
        if not v.strip():
            raise ValueError("Source cannot be empty")
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Clean and validate tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]


class DocumentChunk(BaseModel):
    """Individual document chunk with embeddings."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., min_length=1, description="Chunk content")
    
    # Chunk positioning
    chunk_index: int = Field(..., ge=0, description="Chunk order in document")
    start_char: Optional[int] = Field(None, ge=0, description="Start character position")
    end_char: Optional[int] = Field(None, ge=0, description="End character position")
    
    # Content metadata
    word_count: int = Field(..., ge=0, description="Words in this chunk")
    
    # Vector embeddings (stored as list of floats)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    
    # Relationships
    overlap_with_previous: int = Field(default=0, ge=0, description="Overlap characters with previous chunk")
    overlap_with_next: int = Field(default=0, ge=0, description="Overlap characters with next chunk")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate chunk content."""
        if not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()
    
    @validator('end_char')
    def validate_char_positions(cls, v, values):
        """Ensure end_char > start_char if both are provided."""
        start_char = values.get('start_char')
        if start_char is not None and v is not None and v <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return v


class Document(BaseModel):
    """Complete document representation."""
    
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Processing status
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Chunks
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    
    # Computed properties
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)
    
    @property
    def is_processed(self) -> bool:
        """Check if document is fully processed."""
        return self.status == DocumentStatus.COMPLETED
    
    @property
    def has_embeddings(self) -> bool:
        """Check if chunks have embeddings."""
        return any(chunk.embedding is not None for chunk in self.chunks)
    
    @validator('content')
    def validate_content(cls, v):
        """Validate document content."""
        if not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID."""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the document."""
        if chunk.document_id != self.document_id:
            raise ValueError("Chunk document_id must match document ID")
        self.chunks.append(chunk)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }


# Example usage and validation helpers
def create_document_from_file(file_path: str, content: str, **kwargs) -> Document:
    """Create a document from file path and content."""
    from uuid import uuid4
    
    document_id = str(uuid4())
    file_path_obj = Path(file_path)
    
    # Determine document type from extension
    extension = file_path_obj.suffix.lower()
    doc_type_mapping = {
        '.txt': DocumentType.TEXT,
        '.pdf': DocumentType.PDF,
        '.docx': DocumentType.DOCX,
        '.md': DocumentType.MARKDOWN
    }
    doc_type = doc_type_mapping.get(extension, DocumentType.TEXT)
    
    # Separate metadata-specific kwargs from other kwargs
    metadata_fields = DocumentMetadata.__fields__.keys()
    metadata_kwargs = {k: v for k, v in kwargs.items() if k in metadata_fields}
    
    # Set defaults and create metadata
    metadata_data = {
        'title': kwargs.get('title', file_path_obj.stem),
        'source': file_path,
        'document_type': doc_type,
        'word_count': len(content.split()),
        'file_size': len(content.encode('utf-8')),
        **metadata_kwargs
    }
    
    metadata = DocumentMetadata(**metadata_data)
    
    return Document(
        document_id=document_id,
        content=content,
        metadata=metadata
    )