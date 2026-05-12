"""Document chunking strategies for optimal vector storage."""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re
from uuid import uuid4

from schemas.document import DocumentChunk
from core.logger import app_logger


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"  
    SEMANTIC_SECTIONS = "semantic_sections"
    OVERLAP_SLIDING = "overlap_sliding"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    strategy: ChunkingStrategy = ChunkingStrategy.OVERLAP_SLIDING
    chunk_size: int = 500  # Characters or tokens
    overlap_size: int = 50  # Overlap between chunks
    min_chunk_size: int = 50  # Minimum chunk size
    max_chunk_size: int = 2000  # Maximum chunk size
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True


class DocumentChunker:
    """Intelligent document chunking with multiple strategies."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        app_logger.info(f"Initialized chunker with strategy: {self.config.strategy}")
    
    def chunk_document(
        self, 
        document_id: str, 
        content: str, 
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[DocumentChunk]:
        """Chunk document using specified or default strategy."""
        
        if not content.strip():
            app_logger.warning("Empty content provided for chunking")
            return []
        
        strategy = strategy or self.config.strategy
        
        app_logger.info(f"Chunking document {document_id} with strategy: {strategy}")
        
        try:
            if strategy == ChunkingStrategy.FIXED_SIZE:
                chunks = self._chunk_fixed_size(content)
            elif strategy == ChunkingStrategy.SENTENCE_BASED:
                chunks = self._chunk_by_sentences(content)
            elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
                chunks = self._chunk_by_paragraphs(content)
            elif strategy == ChunkingStrategy.SEMANTIC_SECTIONS:
                chunks = self._chunk_by_semantic_sections(content)
            elif strategy == ChunkingStrategy.OVERLAP_SLIDING:
                chunks = self._chunk_overlap_sliding(content)
            else:
                app_logger.error(f"Unknown chunking strategy: {strategy}")
                return []
            
            # Convert to DocumentChunk objects
            document_chunks = []
            for i, (chunk_content, start_pos, end_pos) in enumerate(chunks):
                if len(chunk_content.strip()) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{i}",
                        document_id=document_id,
                        content=chunk_content.strip(),
                        chunk_index=i,
                        start_char=start_pos,
                        end_char=end_pos,
                        word_count=len(chunk_content.split()),
                        overlap_with_previous=self._calculate_overlap_previous(i, chunks),
                        overlap_with_next=self._calculate_overlap_next(i, chunks)
                    )
                    document_chunks.append(chunk)
            
            app_logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
            return document_chunks
            
        except Exception as e:
            app_logger.error(f"Error chunking document {document_id}: {e}")
            return []
    
    def _chunk_fixed_size(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk content into fixed-size pieces."""
        
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(content), chunk_size):
            start_pos = i
            end_pos = min(i + chunk_size, len(content))
            chunk_content = content[start_pos:end_pos]
            
            # Respect sentence boundaries if configured
            if self.config.respect_sentence_boundaries and end_pos < len(content):
                # Try to end at sentence boundary
                last_sentence_end = max(
                    chunk_content.rfind('.'),
                    chunk_content.rfind('!'),
                    chunk_content.rfind('?')
                )
                if last_sentence_end > chunk_size // 2:  # Only if we're not cutting too much
                    end_pos = start_pos + last_sentence_end + 1
                    chunk_content = content[start_pos:end_pos]
            
            chunks.append((chunk_content, start_pos, end_pos))
        
        return chunks
    
    def _chunk_by_sentences(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk content by sentences, grouping to target size."""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Finalize current chunk
                end_pos = current_start + len(current_chunk)
                chunks.append((current_chunk, current_start, end_pos))
                
                # Start new chunk
                current_chunk = sentence
                current_start = end_pos + 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk if exists
        if current_chunk:
            end_pos = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, min(end_pos, len(content))))
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk content by paragraphs, combining small ones."""
        
        # Split by paragraph breaks
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed max size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Finalize current chunk
                end_pos = current_start + len(current_chunk)
                chunks.append((current_chunk, current_start, end_pos))
                
                # Start new chunk
                current_chunk = paragraph
                current_start = end_pos + 2  # Account for \n\n
            else:
                current_chunk = potential_chunk
        
        # Add final chunk if exists
        if current_chunk:
            end_pos = current_start + len(current_chunk)
            chunks.append((current_chunk, current_start, min(end_pos, len(content))))
        
        return chunks
    
    def _chunk_by_semantic_sections(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk content by semantic sections (headers, topics)."""
        
        # Look for markdown-style headers
        header_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
        matches = list(header_pattern.finditer(content))
        
        if not matches:
            # Fallback to paragraph-based if no headers found
            return self._chunk_by_paragraphs(content)
        
        chunks = []
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Find content until next header or end
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            # If section is too large, sub-chunk it
            if len(section_content) > self.config.max_chunk_size:
                sub_chunks = self._chunk_overlap_sliding(section_content)
                for j, (sub_content, sub_start, sub_end) in enumerate(sub_chunks):
                    chunks.append((sub_content, start_pos + sub_start, start_pos + sub_end))
            else:
                chunks.append((section_content, start_pos, end_pos))
        
        return chunks
    
    def _chunk_overlap_sliding(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk with sliding window and overlap."""
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap_size = self.config.overlap_size
        step_size = chunk_size - overlap_size
        
        for i in range(0, len(content), step_size):
            start_pos = i
            end_pos = min(i + chunk_size, len(content))
            
            # If this would be the last chunk and it's very small, merge with previous
            if end_pos < len(content) and (len(content) - end_pos) < self.config.min_chunk_size:
                end_pos = len(content)
            
            chunk_content = content[start_pos:end_pos]
            
            # Respect sentence boundaries for final adjustment
            if self.config.respect_sentence_boundaries and end_pos < len(content):
                # Look for sentence boundary within reasonable range
                search_start = max(0, len(chunk_content) - overlap_size)
                sentence_boundary = -1
                
                for punct in ['.', '!', '?']:
                    pos = chunk_content.rfind(punct, search_start)
                    if pos > sentence_boundary:
                        sentence_boundary = pos
                
                if sentence_boundary > search_start:
                    end_pos = start_pos + sentence_boundary + 1
                    chunk_content = content[start_pos:end_pos]
            
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunks.append((chunk_content, start_pos, end_pos))
            
            # Break if we've reached the end
            if end_pos >= len(content):
                break
        
        return chunks
    
    def _calculate_overlap_previous(self, chunk_index: int, chunks: List[Tuple[str, int, int]]) -> int:
        """Calculate overlap with previous chunk."""
        if chunk_index == 0:
            return 0
        
        current_start = chunks[chunk_index][1]
        previous_end = chunks[chunk_index - 1][2]
        
        return max(0, previous_end - current_start)
    
    def _calculate_overlap_next(self, chunk_index: int, chunks: List[Tuple[str, int, int]]) -> int:
        """Calculate overlap with next chunk."""
        if chunk_index >= len(chunks) - 1:
            return 0
        
        current_end = chunks[chunk_index][2]
        next_start = chunks[chunk_index + 1][1]
        
        return max(0, current_end - next_start)
    
    def get_optimal_chunk_size(self, content: str) -> int:
        """Suggest optimal chunk size based on content analysis."""
        
        content_length = len(content)
        avg_sentence_length = self._get_average_sentence_length(content)
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Heuristics for optimal chunk size
        if content_length < 1000:
            return min(content_length, 200)
        elif avg_sentence_length > 100:  # Long sentences
            return 800
        elif paragraph_count > 10:  # Many paragraphs
            return 600
        else:
            return 500
    
    def _get_average_sentence_length(self, content: str) -> float:
        """Calculate average sentence length."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0
        
        return sum(len(s) for s in sentences) / len(sentences)


# Example usage and testing
if __name__ == "__main__":
    # Test the chunker
    sample_content = """
    # AI Research Assistant
    
    This is the introduction paragraph that explains what the AI Research Assistant does.
    It provides comprehensive document analysis and question-answering capabilities.
    
    ## Document Processing
    
    The system can process various document types including PDFs, text files, and web articles.
    Each document is processed through multiple stages:
    
    1. Text extraction and cleaning
    2. Intelligent chunking for vector storage
    3. Embedding generation using sentence transformers
    4. Storage in vector database for efficient retrieval
    
    ## Query Processing
    
    When users submit queries, the system performs semantic search across all stored documents.
    It uses advanced ranking algorithms to find the most relevant content.
    The results are then processed by a language model to generate comprehensive answers.
    
    ## Key Features
    
    - Multi-format document support
    - Intelligent text chunking
    - Semantic search capabilities
    - Source attribution and citations
    - Confidence scoring for responses
    """
    
    chunker = DocumentChunker()
    
    # Test different strategies
    for strategy in ChunkingStrategy:
        print(f"\n=== Testing {strategy} ===")
        chunks = chunker.chunk_document("test-doc", sample_content, strategy)
        print(f"Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"\nChunk {i+1} ({len(chunk.content)} chars):")
            print(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)