"""Vector database operations using ChromaDB."""

from typing import List, Dict, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
from pathlib import Path
import uuid
import json
from datetime import datetime

from schemas.document import Document, DocumentChunk
from schemas.query import Query, SearchResult, QueryResult
from storage.embeddings import embedding_manager
from core.config import config
from core.logger import app_logger


class ChromaVectorStore:
    """ChromaDB vector store for document chunks and embeddings."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client and collections.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = Path(persist_directory or config.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._client = None
        self._collection = None
        self._collection_name = "research_documents"
        
        app_logger.info(f"Initialized ChromaVectorStore with persist directory: {self.persist_directory}")
    
    @property
    def client(self) -> chromadb.Client:
        """Lazy load ChromaDB client."""
        if self._client is None:
            app_logger.info("Creating ChromaDB client...")
            
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                app_logger.info("ChromaDB client created successfully")
                
            except Exception as e:
                app_logger.error(f"Failed to create ChromaDB client: {e}")
                raise
        
        return self._client
    
    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the document collection."""
        if self._collection is None:
            try:
                # Try to get existing collection first
                self._collection = self.client.get_collection(
                    name=self._collection_name
                )
                app_logger.info(f"Retrieved existing collection: {self._collection_name}")
                
            except Exception:
                # Collection doesn't exist, create it
                app_logger.info(f"Creating new collection: {self._collection_name}")
                
                try:
                    self._collection = self.client.create_collection(
                        name=self._collection_name,
                        metadata={
                            "description": "AI Research Assistant document chunks",
                            "created_at": datetime.now().isoformat()
                        }
                    )
                    
                    app_logger.info(f"Created new collection: {self._collection_name}")
                    
                except Exception as e:
                    app_logger.error(f"Failed to create collection: {e}")
                    raise
        
        return self._collection
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document and its chunks to the vector store.
        
        Args:
            document: Document object with chunks
            
        Returns:
            Success status
        """
        if not document.chunks:
            app_logger.warning(f"Document {document.document_id} has no chunks to add")
            return False
        
        app_logger.info(f"Adding document {document.document_id} with {len(document.chunks)} chunks")
        
        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in document.chunks]
            embeddings = embedding_manager.generate_embeddings(chunk_texts)
            
            if len(embeddings) != len(document.chunks):
                app_logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(document.chunks)}")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk, embedding in zip(document.chunks, embeddings):
                # Update chunk with embedding
                chunk.embedding = embedding
                
                # Prepare ChromaDB data
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                metadata = {
                    "document_id": document.document_id,
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                    "start_char": chunk.start_char or 0,
                    "end_char": chunk.end_char or 0,
                    
                    # Document metadata
                    "doc_title": document.metadata.title,
                    "doc_author": document.metadata.author or "",
                    "doc_source": document.metadata.source,
                    "doc_type": document.metadata.document_type.value,
                    "doc_created_at": document.metadata.created_at.isoformat(),
                    "doc_tags": json.dumps(document.metadata.tags),
                    "doc_category": document.metadata.category or "",
                    
                    # Processing metadata
                    "ingested_at": datetime.now().isoformat(),
                    "embedding_model": embedding_manager.model_name
                }
                metadatas.append(metadata)
            
            # Ensure collection exists and add to ChromaDB
            collection = self.collection  # This triggers creation if needed
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            app_logger.info(f"Successfully added {len(document.chunks)} chunks for document {document.document_id}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error adding document {document.document_id}: {e}")
            return False
    
    def search_similar(
        self, 
        query_text: str, 
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            where_filter: Metadata filters for ChromaDB
            
        Returns:
            List of SearchResult objects
        """
        if not query_text.strip():
            app_logger.warning("Empty query text provided")
            return []
        
        app_logger.info(f"Searching for similar chunks: '{query_text[:50]}...'")
        
        try:
            # Generate query embedding
            query_embedding = embedding_manager.generate_single_embedding(query_text)
            
            if not query_embedding:
                app_logger.error("Failed to generate query embedding")
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    chunk_id = results['ids'][0][i]
                    document_text = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score (assuming cosine distance)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Create SearchResult
                    search_result = SearchResult(
                        document_id=metadata['document_id'],
                        chunk_id=chunk_id,
                        title=metadata['doc_title'],
                        content_snippet=document_text[:200] + "..." if len(document_text) > 200 else document_text,
                        relevance_score=similarity_score,
                        similarity_score=similarity_score,
                        source=metadata['doc_source'],
                        document_type=metadata['doc_type'],
                        created_at=datetime.fromisoformat(metadata['doc_created_at']),
                        tags=json.loads(metadata.get('doc_tags', '[]'))
                    )
                    
                    search_results.append(search_result)
            
            app_logger.info(f"Found {len(search_results)} similar chunks")
            return search_results
            
        except Exception as e:
            app_logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    chunk_data = {
                        'chunk_id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            app_logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        
        try:
            # Get chunk IDs to delete
            results = self.collection.get(
                where={"document_id": document_id},
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                app_logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                app_logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            app_logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update a specific chunk."""
        
        try:
            # Generate new embedding if content changed
            if not chunk.embedding:
                chunk.embedding = embedding_manager.generate_single_embedding(chunk.content)
            
            # Update in ChromaDB
            self.collection.update(
                ids=[chunk.chunk_id],
                embeddings=[chunk.embedding],
                documents=[chunk.content],
                metadatas=[{
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                    "updated_at": datetime.now().isoformat()
                }]
            )
            
            return True
            
        except Exception as e:
            app_logger.error(f"Error updating chunk {chunk.chunk_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample = self.collection.peek(limit=min(100, count))
            
            # Analyze document types and sources
            doc_types = {}
            sources = set()
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    doc_type = metadata.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    sources.add(metadata.get('doc_source', 'unknown'))
            
            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'document_types': doc_types,
                'collection_name': self._collection_name,
                'persist_directory': str(self.persist_directory),
                'embedding_model': embedding_manager.model_name
            }
            
        except Exception as e:
            app_logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def search_by_metadata(
        self, 
        filters: Dict[str, Any], 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search chunks by metadata filters."""
        
        try:
            results = self.collection.get(
                where=filters,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    chunk_data = {
                        'chunk_id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            app_logger.error(f"Error searching by metadata: {e}")
            return []


# Global vector store instance
vector_store = ChromaVectorStore()

__all__ = ["ChromaVectorStore", "vector_store"]


# Example usage and testing
if __name__ == "__main__":
    from schemas.document import create_document_from_file
    from ingestion.chunking import DocumentChunker
    
    # Test the vector store
    store = ChromaVectorStore()
    
    # Create a sample document
    sample_content = """
    Artificial Intelligence is revolutionizing how we process and understand information.
    Machine learning algorithms can analyze vast datasets to identify patterns and make predictions.
    Natural language processing enables computers to understand and generate human language.
    This technology is being applied in various fields including healthcare, finance, and education.
    """
    
    document = create_document_from_file(
        file_path="sample_ai_doc.txt",
        content=sample_content,
        title="AI Overview Document",
        tags=["AI", "machine-learning", "technology"]
    )
    
    # Chunk the document
    chunker = DocumentChunker()
    document.chunks = chunker.chunk_document(document.document_id, document.content)
    
    print(f"Created document with {len(document.chunks)} chunks")
    
    # Add to vector store
    success = store.add_document(document)
    print(f"Added to vector store: {success}")
    
    # Test search
    results = store.search_similar("machine learning algorithms", n_results=3)
    print(f"\nFound {len(results)} search results:")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result.title} (score: {result.similarity_score:.3f})")
        print(f"   {result.content_snippet}")
    
    # Get collection stats
    stats = store.get_collection_stats()
    print(f"\nCollection stats: {stats}")