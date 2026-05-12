"""Document loader that orchestrates the complete ingestion pipeline."""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import asyncio
from datetime import datetime

from schemas.document import Document, DocumentStatus, create_document_from_file
from ingestion.text_processor import TextProcessor
from ingestion.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
from storage.vector_store import vector_store
from core.logger import app_logger


class DocumentLoader:
    """Orchestrates the complete document ingestion pipeline."""
    
    def __init__(
        self, 
        chunking_config: Optional[ChunkingConfig] = None,
        auto_store: bool = True
    ):
        """
        Initialize document loader.
        
        Args:
            chunking_config: Configuration for document chunking
            auto_store: Whether to automatically store in vector database
        """
        self.text_processor = TextProcessor()
        self.chunker = DocumentChunker(chunking_config)
        self.auto_store = auto_store
        
        app_logger.info("DocumentLoader initialized")
    
    def load_from_text(
        self, 
        content: str, 
        title: str,
        source: str = "direct_input",
        **metadata_kwargs
    ) -> Document:
        """
        Load document from text content.
        
        Args:
            content: Text content to process
            title: Document title
            source: Source identifier
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            Processed Document object
        """
        app_logger.info(f"Loading document from text: '{title}'")
        
        try:
            # Clean and process text
            cleaned_content = self.text_processor.clean_text(content)
            
            if not cleaned_content:
                raise ValueError("Document content is empty after cleaning")
            
            # Extract metadata from content
            extracted_metadata = self.text_processor.extract_metadata_from_text(cleaned_content)
            
            # Merge with provided metadata
            metadata_kwargs.update({
                'word_count': extracted_metadata.get('word_count', 0),
                'language': extracted_metadata.get('estimated_language', 'en')
            })
            
            # Create document
            document = create_document_from_file(
                file_path=source,
                content=cleaned_content,
                title=title,
                **metadata_kwargs
            )
            
            # Process document through pipeline
            return self._process_document(document)
            
        except Exception as e:
            app_logger.error(f"Error loading document from text: {e}")
            raise
    
    def load_from_file(self, file_path: Union[str, Path], **metadata_kwargs) -> Document:
        """
        Load document from file.
        
        Args:
            file_path: Path to the file
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            Processed Document object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        app_logger.info(f"Loading document from file: {file_path}")
        
        try:
            # Read file content based on type
            content = self._read_file_content(file_path)
            
            # Use filename as title if not provided
            title = metadata_kwargs.get('title', file_path.stem)
            
            # Add file-specific metadata
            file_stats = file_path.stat()
            metadata_kwargs.update({
                'file_size': file_stats.st_size,
                'modified_at': datetime.fromtimestamp(file_stats.st_mtime)
            })
            
            # Load using text method
            return self.load_from_text(
                content=content,
                title=title,
                source=str(file_path),
                **metadata_kwargs
            )
            
        except Exception as e:
            app_logger.error(f"Error loading document from file {file_path}: {e}")
            raise
    
    def load_multiple_files(
        self, 
        file_paths: List[Union[str, Path]], 
        **common_metadata
    ) -> List[Document]:
        """
        Load multiple documents from files.
        
        Args:
            file_paths: List of file paths
            **common_metadata: Metadata to apply to all documents
            
        Returns:
            List of processed Document objects
        """
        app_logger.info(f"Loading {len(file_paths)} documents")
        
        documents = []
        successful = 0
        failed = 0
        
        for file_path in file_paths:
            try:
                document = self.load_from_file(file_path, **common_metadata)
                documents.append(document)
                successful += 1
                
            except Exception as e:
                app_logger.error(f"Failed to load {file_path}: {e}")
                failed += 1
        
        app_logger.info(f"Loaded {successful} documents successfully, {failed} failed")
        return documents
    
    async def load_from_text_async(
        self, 
        content: str, 
        title: str,
        source: str = "direct_input",
        **metadata_kwargs
    ) -> Document:
        """Async version of load_from_text."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.load_from_text, 
            content, 
            title, 
            source, 
            **metadata_kwargs
        )
    
    def _process_document(self, document: Document) -> Document:
        """Process document through the complete pipeline."""
        
        try:
            document.status = DocumentStatus.PROCESSING
            
            # Generate chunks
            app_logger.info(f"Chunking document: {document.document_id}")
            chunks = self.chunker.chunk_document(
                document.document_id, 
                document.content
            )
            
            if not chunks:
                raise ValueError("No chunks were generated from document content")
            
            document.chunks = chunks
            document.metadata.chunk_count = len(chunks)
            
            # Store in vector database if configured
            if self.auto_store:
                app_logger.info(f"Storing document in vector database: {document.document_id}")
                success = vector_store.add_document(document)
                
                if not success:
                    raise ValueError("Failed to store document in vector database")
            
            # Mark as completed
            document.status = DocumentStatus.COMPLETED
            
            app_logger.info(f"Successfully processed document: {document.document_id}")
            return document
            
        except Exception as e:
            document.status = DocumentStatus.ERROR
            document.error_message = str(e)
            app_logger.error(f"Error processing document {document.document_id}: {e}")
            raise
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read content from file based on file type."""
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt' or suffix == '.md':
                # Plain text or markdown
                return file_path.read_text(encoding='utf-8')
                
            elif suffix == '.pdf':
                # PDF files (requires pypdf2)
                from pypdf2 import PdfReader
                
                reader = PdfReader(str(file_path))
                content = []
                
                for page in reader.pages:
                    content.append(page.extract_text())
                
                return '\n\n'.join(content)
                
            elif suffix in ['.doc', '.docx']:
                # Word documents (requires python-docx)
                from docx import Document as DocxDocument
                
                doc = DocxDocument(str(file_path))
                content = []
                
                for paragraph in doc.paragraphs:
                    content.append(paragraph.text)
                
                return '\n\n'.join(content)
                
            else:
                # Try to read as text with various encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        return file_path.read_text(encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                
                raise ValueError(f"Could not decode file with any supported encoding")
                
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing pipeline."""
        
        # Get chunker configuration
        chunker_info = {
            'strategy': self.chunker.config.strategy.value,
            'chunk_size': self.chunker.config.chunk_size,
            'overlap_size': self.chunker.config.overlap_size,
            'min_chunk_size': self.chunker.config.min_chunk_size,
            'max_chunk_size': self.chunker.config.max_chunk_size
        }
        
        # Get vector store statistics
        vector_stats = vector_store.get_collection_stats()
        
        return {
            'chunker_config': chunker_info,
            'vector_store_stats': vector_stats,
            'auto_store_enabled': self.auto_store,
            'supported_formats': ['.txt', '.md', '.pdf', '.doc', '.docx']
        }
    
    def update_chunking_strategy(self, strategy: ChunkingStrategy, **config_kwargs):
        """Update the chunking strategy and configuration."""
        
        new_config = ChunkingConfig(strategy=strategy, **config_kwargs)
        self.chunker = DocumentChunker(new_config)
        
        app_logger.info(f"Updated chunking strategy to: {strategy}")
    
    def reprocess_document(self, document: Document) -> Document:
        """Reprocess an existing document with current configuration."""
        
        app_logger.info(f"Reprocessing document: {document.document_id}")
        
        # Clear existing chunks and reset status
        document.chunks = []
        document.status = DocumentStatus.PENDING
        document.error_message = None
        
        # Process again
        return self._process_document(document)


# Global document loader instance
document_loader = DocumentLoader()

__all__ = ["DocumentLoader", "document_loader"]


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Test the document loader
    loader = DocumentLoader()
    
    # Test with sample text
    sample_text = """
    # Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from and make predictions or decisions based on data.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    In supervised learning, the algorithm learns from labeled training data to make 
    predictions on new, unseen data.
    
    ### Unsupervised Learning  
    Unsupervised learning algorithms find patterns in data without explicit labels 
    or target outcomes.
    
    ### Reinforcement Learning
    Reinforcement learning involves an agent learning to make decisions through 
    trial and error interactions with an environment.
    
    ## Applications
    - Image recognition and computer vision
    - Natural language processing and chatbots
    - Recommendation systems
    - Fraud detection
    - Autonomous vehicles
    """
    
    try:
        # Load from text
        print("Loading document from text...")
        document = loader.load_from_text(
            content=sample_text,
            title="Machine Learning Fundamentals",
            tags=["machine-learning", "AI", "fundamentals"],
            category="education"
        )
        
        print(f"✅ Document loaded successfully:")
        print(f"  ID: {document.document_id}")
        print(f"  Status: {document.status}")
        print(f"  Chunks: {len(document.chunks)}")
        print(f"  Word count: {document.metadata.word_count}")
        
        # Show first chunk
        if document.chunks:
            first_chunk = document.chunks[0]
            print(f"\n📄 First chunk preview:")
            print(f"  ID: {first_chunk.chunk_id}")
            print(f"  Words: {first_chunk.word_count}")
            print(f"  Content: {first_chunk.content[:100]}...")
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_file = f.name
        
        print(f"\n📁 Loading from file: {temp_file}")
        file_document = loader.load_from_file(
            temp_file,
            tags=["test", "file-load"]
        )
        
        print(f"✅ File document loaded:")
        print(f"  Title: {file_document.metadata.title}")
        print(f"  Chunks: {len(file_document.chunks)}")
        
        # Get processing stats
        stats = loader.get_processing_stats()
        print(f"\n📊 Processing stats:")
        print(f"  Chunking strategy: {stats['chunker_config']['strategy']}")
        print(f"  Total chunks in DB: {stats['vector_store_stats']['total_chunks']}")
        print(f"  Supported formats: {stats['supported_formats']}")
        
        # Clean up
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        app_logger.error(f"Document loader test failed: {e}")