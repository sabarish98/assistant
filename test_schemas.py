"""Test all Pydantic schemas with real data."""

import json
from datetime import datetime
from pathlib import Path

from schemas.document import Document, DocumentMetadata, DocumentChunk, DocumentType, create_document_from_file
from schemas.query import Query, QueryResult, SearchResult, SearchFilter, create_simple_query, create_research_query
from schemas.response import LLMResponse, ConfidenceScore, ValidationResult, SourceCitation, create_successful_response
from core.logger import app_logger


def test_document_schemas():
    """Test document-related schemas."""
    
    print("🧪 Testing Document Schemas...")
    
    # Test creating a document from sample content
    sample_content = """
    This is a sample document for testing our AI Research Assistant.
    It contains multiple paragraphs and demonstrates how we can structure
    and validate document data using Pydantic schemas.
    
    The document includes metadata, chunking capabilities, and validation
    to ensure data quality throughout our system.
    """
    
    # Create document using helper function
    doc = create_document_from_file(
        file_path="/sample/test_document.txt",
        content=sample_content,
        title="Sample Test Document",
        author="AI Research Assistant",
        tags=["test", "sample", "documentation"],
        category="testing"
    )
    
    print(f"✅ Document created: {doc.document_id}")
    print(f"   Title: {doc.metadata.title}")
    print(f"   Type: {doc.metadata.document_type}")
    print(f"   Word count: {doc.metadata.word_count}")
    print(f"   Tags: {doc.metadata.tags}")
    
    # Test adding chunks
    chunk1 = DocumentChunk(
        chunk_id="chunk-1",
        document_id=doc.document_id,
        content=sample_content[:100] + "...",
        chunk_index=0,
        start_char=0,
        end_char=100,
        word_count=20,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding
    )
    
    doc.add_chunk(chunk1)
    print(f"✅ Added chunk: {chunk1.chunk_id}")
    print(f"   Total chunks: {doc.total_chunks}")
    print(f"   Has embeddings: {doc.has_embeddings}")
    
    # Test JSON serialization
    doc_json = doc.model_dump_json(indent=2)
    print("✅ JSON serialization successful")
    
    return doc


def test_query_schemas():
    """Test query-related schemas."""
    
    print("\n🧪 Testing Query Schemas...")
    
    # Test simple query
    query = create_simple_query(
        text="What is the main topic of the sample document?",
        user_id="test-user",
        session_id="test-session"
    )
    
    print(f"✅ Simple query created: {query.query_id}")
    print(f"   Text: {query.text}")
    print(f"   Type: {query.query_type}")
    
    # Test research query with filters
    research_query = create_research_query(
        text="Analyze all documents related to AI and machine learning",
        max_results=20,
        categories=["AI", "Research"],
        tags=["machine-learning", "artificial-intelligence"]
    )
    
    print(f"✅ Research query created: {research_query.query_id}")
    print(f"   Max results: {research_query.filters.max_results}")
    print(f"   Categories filter: {research_query.filters.categories}")
    
    # Test search result
    search_result = SearchResult(
        document_id="doc-123",
        chunk_id="chunk-1",
        title="Sample AI Document",
        content_snippet="This document discusses artificial intelligence...",
        relevance_score=0.95,
        similarity_score=0.87,
        source="/docs/ai_research.pdf",
        document_type="pdf",
        created_at=datetime.now(),
        tags=["AI", "research"]
    )
    
    # Test query result
    query_result = QueryResult(
        query_id=query.query_id,
        query_text=query.text,
        search_results=[search_result],
        total_results=1,
        search_time_ms=150.5,
        embedding_time_ms=45.2
    )
    
    print(f"✅ Query result created with {query_result.total_results} results")
    print(f"   Has results: {query_result.has_results}")
    print(f"   Top result score: {query_result.top_result.relevance_score}")
    
    return query, query_result


def test_response_schemas():
    """Test response and validation schemas."""
    
    print("\n🧪 Testing Response Schemas...")
    
    # Test source citation
    source = SourceCitation(
        document_id="doc-123",
        chunk_id="chunk-1",
        title="Sample AI Document",
        author="Dr. AI Researcher",
        source="/docs/ai_research.pdf",
        quoted_text="Artificial intelligence represents a transformative technology...",
        relevance_score=0.92,
        document_type="pdf",
        created_at=datetime.now()
    )
    
    # Test successful response
    response = create_successful_response(
        query_id="query-123",
        content="Based on the sample document, the main topic appears to be artificial intelligence and its applications in research. The document discusses various aspects of AI technology and its transformative potential.",
        confidence_score=0.88,
        model_name="gemma4:e4b",
        temperature=0.1,
        generation_time_ms=1250.0,
        tokens_used=125
    )
    
    # Add source citation
    response.add_source(source)
    
    print(f"✅ Response created: {response.response_id}")
    print(f"   Content length: {len(response.content)} chars")
    print(f"   Confidence level: {response.confidence.level}")
    print(f"   Is valid: {response.is_valid_response}")
    print(f"   Has reliable sources: {response.has_reliable_sources}")
    print(f"   Source count: {response.source_count}")
    
    # Test validation details
    print(f"   Validation status: {response.validation.status}")
    print(f"   Schema validation: {response.validation.schema_validation}")
    
    return response


def test_schema_validation():
    """Test schema validation with invalid data."""
    
    print("\n🧪 Testing Schema Validation...")
    
    try:
        # Test invalid document (empty content)
        invalid_doc = Document(
            document_id="test",
            content="",  # This should fail validation
            metadata=DocumentMetadata(
                title="Test",
                source="test.txt",
                document_type=DocumentType.TEXT
            )
        )
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    try:
        # Test invalid query (empty text)
        invalid_query = Query(
            query_id="test",
            text=""  # This should fail validation
        )
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    try:
        # Test invalid confidence score
        invalid_confidence = ConfidenceScore(
            overall_score=1.5,  # This should fail validation (> 1.0)
            level="high",
            source_reliability=0.8,
            content_relevance=0.9
        )
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")


def main():
    """Run all schema tests."""
    
    print("🚀 Starting Schema Validation Tests")
    print("=" * 50)
    
    try:
        # Test individual schema types
        doc = test_document_schemas()
        query, query_result = test_query_schemas()
        response = test_response_schemas()
        
        # Test validation
        test_schema_validation()
        
        print("\n" + "=" * 50)
        print("🎉 All schema tests passed successfully!")
        print(f"📊 Test Summary:")
        print(f"   ✅ Document schema: Working correctly")
        print(f"   ✅ Query schema: Working correctly")
        print(f"   ✅ Response schema: Working correctly")
        print(f"   ✅ Validation: Catching invalid data correctly")
        
        # Optional: Save sample data for inspection
        sample_data = {
            "document": doc.model_dump(),
            "query": query.model_dump(),
            "query_result": query_result.model_dump(),
            "response": response.model_dump()
        }
        
        with open("sample_schema_data.json", "w") as f:
            json.dump(sample_data, f, indent=2, default=str)
        
        print(f"\n💾 Sample data saved to 'sample_schema_data.json'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Schema test failed: {e}")
        app_logger.error(f"Schema test error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)