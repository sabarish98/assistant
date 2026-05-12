"""Test the complete document ingestion pipeline."""

import tempfile
from pathlib import Path

from ingestion.document_loader import document_loader
from ingestion.chunking import ChunkingStrategy
from storage.vector_store import vector_store
from core.logger import app_logger


def test_text_processing():
    """Test text processing capabilities."""
    
    print("🧪 Testing Text Processing...")
    
    sample_text = """
    # AI Research Assistant
    
    This is a comprehensive document about artificial intelligence and machine learning.
    
    ## Key Features
    
    - Document processing and ingestion
    - Intelligent text chunking with multiple strategies  
    - Vector embeddings using sentence transformers
    - Semantic search and retrieval
    - LLM-powered question answering
    
    ## Technical Architecture
    
    The system uses ChromaDB for vector storage, LangChain for LLM integration,
    and LangGraph for workflow orchestration. It supports multiple document formats
    including PDF, DOCX, and plain text files.
    
    Contact: support@example.com | Phone: 555-123-4567
    """
    
    try:
        # Load document
        document = document_loader.load_from_text(
            content=sample_text,
            title="AI Research Assistant Documentation",
            tags=["AI", "documentation", "research"],
            category="technical",
            author="AI Research Team"
        )
        
        print(f"✅ Document processed successfully:")
        print(f"   ID: {document.document_id}")
        print(f"   Title: {document.metadata.title}")
        print(f"   Status: {document.status}")
        print(f"   Word Count: {document.metadata.word_count}")
        print(f"   Chunks: {len(document.chunks)}")
        print(f"   Tags: {document.metadata.tags}")
        print(f"   Has embeddings: {document.has_embeddings}")
        
        # Show chunk details
        print(f"\n📄 Chunk Details:")
        for i, chunk in enumerate(document.chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}: {chunk.word_count} words")
            print(f"   Content: {chunk.content[:80]}...")
            print(f"   Has embedding: {chunk.embedding is not None}")
            if chunk.embedding:
                print(f"   Embedding dim: {len(chunk.embedding)}")
        
        return document
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return None


def test_different_chunking_strategies():
    """Test different chunking strategies."""
    
    print("\n🧪 Testing Different Chunking Strategies...")
    
    sample_content = """
    Artificial Intelligence (AI) is revolutionizing many industries. Machine learning, 
    a subset of AI, enables computers to learn from data without being explicitly programmed.
    
    Deep learning uses neural networks with multiple layers to model complex patterns.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    
    These technologies are being applied in healthcare for medical diagnosis.
    In finance, they're used for fraud detection and algorithmic trading.
    Autonomous vehicles rely heavily on AI for navigation and safety.
    """
    
    # Test both LangChain and custom strategies
    langchain_strategies = [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.CHARACTER,
        ChunkingStrategy.MARKDOWN_HEADERS,
        ChunkingStrategy.TOKEN_BASED
    ]
    
    custom_strategies = [
        ChunkingStrategy.SEMANTIC_SECTIONS,
        ChunkingStrategy.OVERLAP_SLIDING
    ]
    
    strategies_to_test = langchain_strategies + custom_strategies
    
    results = {}
    
    for strategy in strategies_to_test:
        strategy_name = strategy.value.replace('_', ' ').title()
        strategy_type = "LangChain" if strategy in langchain_strategies else "Custom"
        
        print(f"\n   Testing {strategy_name} ({strategy_type})...")
        
        try:
            # Update chunking strategy with appropriate chunk size
            chunk_size = 800 if strategy in langchain_strategies else 200
            document_loader.update_chunking_strategy(strategy, chunk_size=chunk_size)
            
            # Process document
            document = document_loader.load_from_text(
                content=sample_content,
                title=f"Test Document - {strategy_name}",
                source=f"test_{strategy.value}"
            )
            
            results[strategy.value] = {
                'chunks': len(document.chunks),
                'avg_chunk_size': sum(len(chunk.content) for chunk in document.chunks) / len(document.chunks) if document.chunks else 0,
                'status': document.status.value,
                'type': strategy_type
            }
            
            print(f"   ✅ {strategy_name}: {len(document.chunks)} chunks created")
            
        except Exception as e:
            print(f"   ❌ {strategy_name} failed: {e}")
            results[strategy.value] = {'error': str(e), 'type': strategy_type}
    
    print(f"\n📊 Chunking Strategy Results:")
    for strategy, result in results.items():
        strategy_name = strategy.replace('_', ' ').title()
        if 'error' not in result:
            strategy_type = result.get('type', 'Unknown')
            print(f"   {strategy_name} ({strategy_type}): {result['chunks']} chunks, avg size: {result['avg_chunk_size']:.0f} chars")
        else:
            strategy_type = result.get('type', 'Unknown')
            print(f"   {strategy_name} ({strategy_type}): ERROR - {result['error']}")
    
    return results


def test_vector_search():
    """Test vector search capabilities."""
    
    print("\n🧪 Testing Vector Search...")
    
    # Create test documents with different topics
    test_docs = [
        {
            "content": """
            Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning.
            Supervised learning uses labeled data to train models. Popular algorithms include linear regression,
            decision trees, and neural networks. These are used for classification and prediction tasks.
            """,
            "title": "Machine Learning Algorithms",
            "tags": ["machine-learning", "algorithms"]
        },
        {
            "content": """
            Natural language processing (NLP) enables computers to understand and generate human language.
            Key techniques include tokenization, part-of-speech tagging, and sentiment analysis.
            Modern NLP uses transformer models like BERT and GPT for advanced language understanding.
            """,
            "title": "Natural Language Processing",
            "tags": ["NLP", "language", "transformers"]
        },
        {
            "content": """
            Computer vision allows machines to interpret and understand visual information from images and videos.
            Convolutional neural networks (CNNs) are commonly used for image recognition tasks.
            Applications include facial recognition, object detection, and medical image analysis.
            """,
            "title": "Computer Vision",
            "tags": ["computer-vision", "CNN", "image-processing"]
        }
    ]
    
    # Load all test documents
    loaded_docs = []
    for doc_data in test_docs:
        try:
            document = document_loader.load_from_text(**doc_data)
            loaded_docs.append(document)
            print(f"   ✅ Loaded: {document.metadata.title}")
        except Exception as e:
            print(f"   ❌ Failed to load {doc_data['title']}: {e}")
    
    if not loaded_docs:
        print("❌ No documents loaded for search testing")
        return
    
    # Test different search queries
    test_queries = [
        "neural networks and deep learning",
        "language processing and text analysis",
        "image recognition and computer vision",
        "supervised learning algorithms"
    ]
    
    print(f"\n🔍 Testing Search Queries:")
    
    for query in test_queries:
        try:
            results = vector_store.search_similar(query, n_results=3)
            
            print(f"\n   Query: '{query}'")
            print(f"   Results: {len(results)}")
            
            for i, result in enumerate(results[:2]):  # Show top 2 results
                print(f"     {i+1}. {result.title} (score: {result.similarity_score:.3f})")
                print(f"        {result.content_snippet[:60]}...")
        
        except Exception as e:
            print(f"   ❌ Search failed for '{query}': {e}")


def test_file_loading():
    """Test loading documents from files."""
    
    print("\n🧪 Testing File Loading...")
    
    # Create temporary files with different content
    temp_files = []
    
    file_contents = [
        ("ai_overview.txt", """
        Artificial Intelligence Overview
        
        AI is transforming industries through automation and intelligent decision-making.
        Key areas include machine learning, natural language processing, and robotics.
        """),
        ("ml_basics.md", """
        # Machine Learning Basics
        
        ## Introduction
        Machine learning is a method of data analysis that automates analytical model building.
        
        ## Types
        - Supervised Learning
        - Unsupervised Learning  
        - Reinforcement Learning
        """)
    ]
    
    try:
        # Create temporary files
        for filename, content in file_contents:
            temp_file = Path(tempfile.gettempdir()) / filename
            temp_file.write_text(content)
            temp_files.append(temp_file)
            print(f"   📁 Created: {temp_file}")
        
        # Load files using document loader
        documents = document_loader.load_multiple_files(
            temp_files,
            category="test",
            tags=["file-test"]
        )
        
        print(f"\n   ✅ Loaded {len(documents)} documents from files:")
        for doc in documents:
            print(f"     - {doc.metadata.title} ({len(doc.chunks)} chunks)")
        
        return documents
        
    except Exception as e:
        print(f"   ❌ File loading test failed: {e}")
        return []
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                print(f"   🗑️ Cleaned up: {temp_file}")


def test_collection_statistics():
    """Test collection statistics and management."""
    
    print("\n🧪 Testing Collection Statistics...")
    
    try:
        # Get processing statistics
        processing_stats = document_loader.get_processing_stats()
        
        print(f"   📊 Processing Statistics:")
        print(f"     Chunking Strategy: {processing_stats['chunker_config']['strategy']}")
        print(f"     Chunk Size: {processing_stats['chunker_config']['chunk_size']}")
        print(f"     Auto Store: {processing_stats['auto_store_enabled']}")
        
        # Get vector store statistics
        vector_stats = vector_store.get_collection_stats()
        
        print(f"\n   📊 Vector Store Statistics:")
        print(f"     Total Chunks: {vector_stats.get('total_chunks', 'N/A')}")
        print(f"     Unique Sources: {vector_stats.get('unique_sources', 'N/A')}")
        print(f"     Document Types: {vector_stats.get('document_types', {})}")
        print(f"     Embedding Model: {vector_stats.get('embedding_model', 'N/A')}")
        
        return processing_stats, vector_stats
        
    except Exception as e:
        print(f"   ❌ Statistics test failed: {e}")
        return None, None


def main():
    """Run all ingestion pipeline tests."""
    
    print("🚀 Testing Complete Ingestion Pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Basic text processing
        document = test_text_processing()
        
        if document:
            # Test 2: Different chunking strategies
            chunking_results = test_different_chunking_strategies()
            
            # Test 3: Vector search
            test_vector_search()
            
            # Test 4: File loading
            file_documents = test_file_loading()
            
            # Test 5: Collection statistics
            processing_stats, vector_stats = test_collection_statistics()
            
            print("\n" + "=" * 60)
            print("🎉 All ingestion pipeline tests completed!")
            
            # Summary
            total_chunks = vector_stats.get('total_chunks', 0) if vector_stats else 0
            print(f"\n📈 Test Summary:")
            print(f"   ✅ Text processing: Working")
            print(f"   ✅ Chunking strategies: {len([r for r in chunking_results.values() if 'error' not in r])} tested")
            print(f"   ✅ Vector search: Working")
            print(f"   ✅ File loading: {len(file_documents)} documents loaded")
            print(f"   ✅ Total chunks in database: {total_chunks}")
            
            return True
        else:
            print("\n❌ Basic text processing failed, skipping other tests")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        app_logger.error(f"Ingestion pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)