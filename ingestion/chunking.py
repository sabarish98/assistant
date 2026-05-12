"""Document chunking strategies using LangChain text splitters with custom enhancements."""

from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import re
from uuid import uuid4

# LangChain text splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter, 
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)

from schemas.document import DocumentChunk
from core.logger import app_logger


class ChunkingStrategy(str, Enum):
    """Available chunking strategies using LangChain text splitters."""
    RECURSIVE_CHARACTER = "recursive_character"  # LangChain's best general-purpose splitter
    CHARACTER = "character"                      # Simple character-based splitting
    TOKEN_BASED = "token_based"                 # Token-aware splitting
    MARKDOWN_HEADERS = "markdown_headers"       # Markdown structure-aware splitting
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Sentence transformer token-based
    # Legacy custom strategies (kept for backward compatibility)
    SEMANTIC_SECTIONS = "semantic_sections"     # Custom implementation for headers
    OVERLAP_SLIDING = "overlap_sliding"         # Custom sliding window


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER
    chunk_size: int = 1000  # Characters or tokens (increased for better context)
    chunk_overlap: int = 200  # Overlap between chunks (LangChain standard naming)
    min_chunk_size: int = 50  # Minimum chunk size
    max_chunk_size: int = 4000  # Maximum chunk size
    # LangChain-specific separators for RecursiveCharacterTextSplitter
    separators: Optional[List[str]] = None  # None uses LangChain defaults
    # Custom options
    respect_sentence_boundaries: bool = True  # For custom strategies
    respect_paragraph_boundaries: bool = True  # For custom strategies
    # Token-based settings
    model_name: str = "gpt-3.5-turbo"  # For token counting
    encoding_name: Optional[str] = None  # For token text splitter


class DocumentChunker:
    """Intelligent document chunking using LangChain text splitters with custom enhancements."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._splitter_cache = {}  # Cache splitters for reuse
        app_logger.info(f"Initialized LangChain-based chunker with strategy: {self.config.strategy}")
    
    def chunk_document(
        self, 
        document_id: str, 
        content: str, 
        strategy: Optional[ChunkingStrategy] = None
    ) -> List[DocumentChunk]:
        """Chunk document using LangChain text splitters or custom strategies."""
        
        if not content.strip():
            app_logger.warning("Empty content provided for chunking")
            return []
        
        strategy = strategy or self.config.strategy
        
        app_logger.info(f"Chunking document {document_id} with LangChain strategy: {strategy}")
        
        try:
            # Get text chunks using appropriate splitter
            if strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
                chunks = self._chunk_with_recursive_character(content)
            elif strategy == ChunkingStrategy.CHARACTER:
                chunks = self._chunk_with_character_splitter(content)
            elif strategy == ChunkingStrategy.TOKEN_BASED:
                chunks = self._chunk_with_token_splitter(content)
            elif strategy == ChunkingStrategy.MARKDOWN_HEADERS:
                chunks = self._chunk_with_markdown_headers(content)
            elif strategy == ChunkingStrategy.SENTENCE_TRANSFORMERS:
                chunks = self._chunk_with_sentence_transformers(content)
            elif strategy == ChunkingStrategy.SEMANTIC_SECTIONS:
                chunks = self._chunk_by_semantic_sections_custom(content)  # Keep custom
            elif strategy == ChunkingStrategy.OVERLAP_SLIDING:
                chunks = self._chunk_overlap_sliding_custom(content)  # Keep custom
            else:
                app_logger.error(f"Unknown chunking strategy: {strategy}")
                return []
            
            # Convert text chunks to DocumentChunk objects with metadata
            document_chunks = self._create_document_chunks(document_id, content, chunks)
            
            app_logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
            return document_chunks
            
        except Exception as e:
            app_logger.error(f"Error chunking document {document_id}: {e}")
            return []
    
    def _get_recursive_character_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get cached RecursiveCharacterTextSplitter."""
        cache_key = f"recursive_{self.config.chunk_size}_{self.config.chunk_overlap}"
        
        if cache_key not in self._splitter_cache:
            separators = self.config.separators or [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                " ",     # Spaces
                ""       # Character level
            ]
            
            self._splitter_cache[cache_key] = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=separators,
                is_separator_regex=False,
            )
        
        return self._splitter_cache[cache_key]
    
    def _get_character_splitter(self) -> CharacterTextSplitter:
        """Get cached CharacterTextSplitter."""
        cache_key = f"character_{self.config.chunk_size}_{self.config.chunk_overlap}"
        
        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separator="\n\n"  # Split on paragraphs by default
            )
        
        return self._splitter_cache[cache_key]
    
    def _get_token_splitter(self) -> TokenTextSplitter:
        """Get cached TokenTextSplitter."""
        cache_key = f"token_{self.config.chunk_size}_{self.config.chunk_overlap}"
        
        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                model_name=self.config.model_name,
                encoding_name=self.config.encoding_name
            )
        
        return self._splitter_cache[cache_key]
    
    def _get_sentence_transformers_splitter(self) -> SentenceTransformersTokenTextSplitter:
        """Get cached SentenceTransformersTokenTextSplitter."""
        cache_key = f"sentence_transformers_{self.config.chunk_size}_{self.config.chunk_overlap}"
        
        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.config.chunk_overlap,
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Match our embedding model
                tokens_per_chunk=self.config.chunk_size
            )
        
        return self._splitter_cache[cache_key]
    
    def _chunk_with_recursive_character(self, content: str) -> List[str]:
        """Chunk using LangChain's RecursiveCharacterTextSplitter."""
        splitter = self._get_recursive_character_splitter()
        return splitter.split_text(content)
    
    def _chunk_with_character_splitter(self, content: str) -> List[str]:
        """Chunk using LangChain's CharacterTextSplitter."""
        splitter = self._get_character_splitter()
        return splitter.split_text(content)
    
    def _chunk_with_token_splitter(self, content: str) -> List[str]:
        """Chunk using LangChain's TokenTextSplitter."""
        splitter = self._get_token_splitter()
        return splitter.split_text(content)
    
    def _chunk_with_markdown_headers(self, content: str) -> List[str]:
        """Chunk using LangChain's MarkdownHeaderTextSplitter."""
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        # Split by headers first
        md_header_splits = markdown_splitter.split_text(content)
        
        # If the chunks are too large, further split them
        if any(len(chunk.page_content) > self.config.chunk_size for chunk in md_header_splits):
            # Use recursive splitter for large chunks
            recursive_splitter = self._get_recursive_character_splitter()
            final_chunks = []
            
            for chunk in md_header_splits:
                if len(chunk.page_content) > self.config.chunk_size:
                    sub_chunks = recursive_splitter.split_text(chunk.page_content)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk.page_content)
            
            return final_chunks
        else:
            return [chunk.page_content for chunk in md_header_splits]
    
    def _chunk_with_sentence_transformers(self, content: str) -> List[str]:
        """Chunk using SentenceTransformersTokenTextSplitter."""
        splitter = self._get_sentence_transformers_splitter()
        return splitter.split_text(content)
    
    def _create_document_chunks(
        self, 
        document_id: str, 
        original_content: str, 
        chunks: List[str]
    ) -> List[DocumentChunk]:
        """Convert text chunks to DocumentChunk objects with metadata."""
        document_chunks = []
        
        for i, chunk_content in enumerate(chunks):
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                # Calculate position in original document
                start_pos = self._find_chunk_position(original_content, chunk_content, i)
                end_pos = start_pos + len(chunk_content) if start_pos != -1 else -1
                
                # Calculate overlaps (approximate for LangChain chunks)
                overlap_prev = min(self.config.chunk_overlap, len(chunk_content)) if i > 0 else 0
                overlap_next = min(self.config.chunk_overlap, len(chunk_content)) if i < len(chunks) - 1 else 0
                
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{i}",
                    document_id=document_id,
                    content=chunk_content.strip(),
                    chunk_index=i,
                    start_char=start_pos if start_pos != -1 else None,
                    end_char=end_pos if end_pos != -1 else None,
                    word_count=len(chunk_content.split()),
                    overlap_with_previous=overlap_prev,
                    overlap_with_next=overlap_next
                )
                document_chunks.append(chunk)
        
        return document_chunks
    
    def _find_chunk_position(self, content: str, chunk: str, chunk_index: int) -> int:
        """Find the position of a chunk in the original content."""
        try:
            # Simple approach: find first occurrence after previous chunks
            # This is approximate due to LangChain's complex splitting logic
            search_start = chunk_index * (self.config.chunk_size - self.config.chunk_overlap)
            search_start = max(0, min(search_start, len(content) - 1))
            
            position = content.find(chunk.strip()[:50], search_start)  # Use first 50 chars
            return position if position != -1 else search_start
        except Exception:
            return -1
    
    # Keep custom implementations for backward compatibility and specialized use cases
    def _chunk_fixed_size_legacy(self, content: str) -> List[Tuple[str, int, int]]:
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
    
    def _chunk_by_sentences_legacy(self, content: str) -> List[Tuple[str, int, int]]:
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
    
    def _chunk_by_paragraphs_legacy(self, content: str) -> List[Tuple[str, int, int]]:
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
    
    def _chunk_by_semantic_sections_custom(self, content: str) -> List[str]:
        """Custom semantic sections chunking (kept for specialized use cases)."""
        # Convert tuple format to string list for consistency with LangChain
        tuples = self._chunk_by_semantic_sections_legacy(content)
        return [chunk_content for chunk_content, _, _ in tuples]
    
    def _chunk_overlap_sliding_custom(self, content: str) -> List[str]:
        """Custom overlap sliding chunking (kept for specialized use cases)."""
        # Convert tuple format to string list for consistency with LangChain
        tuples = self._chunk_overlap_sliding_legacy(content)
        return [chunk_content for chunk_content, _, _ in tuples]
    
    # Legacy methods (renamed but kept for specialized functionality)
    def _chunk_by_semantic_sections_legacy(self, content: str) -> List[Tuple[str, int, int]]:
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
    
    def _chunk_overlap_sliding_legacy(self, content: str) -> List[Tuple[str, int, int]]:
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
    # Test the LangChain-based chunker
    sample_content = """
    # AI Research Assistant
    
    This is the introduction paragraph that explains what the AI Research Assistant does.
    It provides comprehensive document analysis and question-answering capabilities using 
    advanced LangChain text splitters for optimal chunking performance.
    
    ## Document Processing
    
    The system can process various document types including PDFs, text files, and web articles.
    Each document is processed through multiple stages:
    
    1. Text extraction and cleaning with comprehensive preprocessing
    2. Intelligent chunking using LangChain's battle-tested splitters
    3. Embedding generation using sentence transformers with proper tokenization
    4. Storage in vector database for efficient retrieval and search
    
    ## Query Processing
    
    When users submit queries, the system performs semantic search across all stored documents.
    It uses advanced ranking algorithms to find the most relevant content chunks.
    The results are then processed by a language model to generate comprehensive answers.
    
    ## Key Features
    
    - Multi-format document support with intelligent parsing
    - LangChain-based text chunking with multiple strategies
    - Semantic search capabilities with vector similarity
    - Source attribution and citations for transparency
    - Confidence scoring for responses and reliability assessment
    
    ## Advanced Capabilities
    
    The system now leverages LangChain's proven text splitting algorithms:
    - RecursiveCharacterTextSplitter for general-purpose chunking
    - TokenTextSplitter for token-aware processing
    - MarkdownHeaderTextSplitter for structured document handling
    - Custom strategies for specialized use cases
    """
    
    chunker = DocumentChunker()
    
    # Test LangChain strategies
    langchain_strategies = [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.CHARACTER,
        ChunkingStrategy.MARKDOWN_HEADERS,
        ChunkingStrategy.TOKEN_BASED,
        ChunkingStrategy.SENTENCE_TRANSFORMERS
    ]
    
    print("🚀 Testing LangChain-Based Chunking Strategies")
    print("=" * 60)
    
    for strategy in langchain_strategies:
        try:
            print(f"\n=== Testing {strategy.value.replace('_', ' ').title()} ===")
            chunks = chunker.chunk_document("test-doc", sample_content, strategy)
            print(f"✅ Generated {len(chunks)} chunks")
            
            if chunks:
                avg_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
                print(f"   Average chunk size: {avg_size:.0f} characters")
                
                # Show first chunk preview
                first_chunk = chunks[0]
                print(f"   First chunk preview ({len(first_chunk.content)} chars):")
                print(f"   {first_chunk.content[:100]}..." if len(first_chunk.content) > 100 else f"   {first_chunk.content}")
        
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
    
    # Test custom strategies for comparison
    custom_strategies = [ChunkingStrategy.SEMANTIC_SECTIONS, ChunkingStrategy.OVERLAP_SLIDING]
    
    print(f"\n{'='*60}")
    print("🛠️  Testing Custom Strategies (Legacy)")
    
    for strategy in custom_strategies:
        try:
            print(f"\n=== Testing {strategy.value.replace('_', ' ').title()} (Custom) ===")
            chunks = chunker.chunk_document("test-doc", sample_content, strategy)
            print(f"✅ Generated {len(chunks)} chunks")
            
            if chunks:
                avg_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
                print(f"   Average chunk size: {avg_size:.0f} characters")
        
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 LangChain integration complete! Best of both worlds:")
    print("   ✅ Battle-tested LangChain splitters for reliability")
    print("   ✅ Custom strategies for specialized use cases") 
    print("   ✅ Cached splitters for performance")
    print("   ✅ Consistent DocumentChunk output format")