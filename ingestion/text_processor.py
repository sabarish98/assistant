"""Text processing utilities for document ingestion."""

import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from core.logger import app_logger


class TextProcessor:
    """Text cleaning and processing utilities."""
    
    def __init__(self):
        # Common patterns for cleaning
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s.,!?;:()\[\]"-]')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """Clean and normalize text content."""
        
        if not text:
            return ""
        
        # Remove null characters and control characters
        text = text.replace('\x00', '').replace('\r', '\n')
        
        # Normalize whitespace
        if preserve_structure:
            # Keep paragraph breaks but normalize other whitespace
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        else:
            # Normalize all whitespace to single spaces
            text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_metadata_from_text(self, text: str) -> Dict[str, any]:
        """Extract basic metadata from text content."""
        
        metadata = {}
        
        # Basic statistics
        metadata['character_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['line_count'] = text.count('\n') + 1
        
        # Paragraph estimation (double newlines)
        metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Find emails and URLs
        emails = self.email_pattern.findall(text)
        urls = self.url_pattern.findall(text)
        
        metadata['contains_emails'] = len(emails) > 0
        metadata['contains_urls'] = len(urls) > 0
        metadata['email_count'] = len(emails)
        metadata['url_count'] = len(urls)
        
        # Language detection (basic)
        metadata['estimated_language'] = self._detect_language_simple(text)
        
        # Structure analysis
        metadata['has_bullet_points'] = bool(re.search(r'^\s*[•\-\*]\s', text, re.MULTILINE))
        metadata['has_numbers'] = bool(re.search(r'\d', text))
        metadata['has_headings'] = bool(re.search(r'^#+\s', text, re.MULTILINE))  # Markdown headings
        
        return metadata
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on common words."""
        
        text_lower = text.lower()
        
        # English indicators
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Spanish indicators
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        # French indicators
        french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
        french_count = sum(1 for word in french_words if word in text_lower)
        
        # Simple majority voting
        if english_count >= spanish_count and english_count >= french_count:
            return 'en'
        elif spanish_count >= french_count:
            return 'es'
        elif french_count > 0:
            return 'fr'
        else:
            return 'unknown'
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract sections from structured text."""
        
        sections = []
        
        # Try to find markdown-style headers
        header_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
        matches = list(header_pattern.finditer(text))
        
        if matches:
            # Process markdown sections
            for i, match in enumerate(matches):
                level = len(match.group(1))
                title = match.group(2).strip()
                start_pos = match.end()
                
                # Find content until next header or end
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                content = text[start_pos:end_pos].strip()
                
                sections.append({
                    'level': level,
                    'title': title,
                    'content': content,
                    'start_pos': match.start(),
                    'end_pos': end_pos
                })
        else:
            # Try paragraph-based sections
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                sections.append({
                    'level': 1,
                    'title': f"Section {i+1}",
                    'content': paragraph,
                    'start_pos': 0,  # Would need more complex calculation
                    'end_pos': len(paragraph)
                })
        
        return sections
    
    def remove_pii_basic(self, text: str) -> Tuple[str, List[str]]:
        """Basic PII removal (emails, phone numbers, etc.)."""
        
        removed_items = []
        cleaned_text = text
        
        # Remove emails
        emails = self.email_pattern.findall(text)
        for email in emails:
            cleaned_text = cleaned_text.replace(email, '[EMAIL_REMOVED]')
            removed_items.append(f"Email: {email}")
        
        # Remove phone numbers (basic patterns)
        phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        phones = phone_pattern.findall(text)
        for phone in phones:
            cleaned_text = cleaned_text.replace(phone, '[PHONE_REMOVED]')
            removed_items.append(f"Phone: {phone}")
        
        # Remove SSN-like patterns (basic)
        ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        ssns = ssn_pattern.findall(text)
        for ssn in ssns:
            cleaned_text = cleaned_text.replace(ssn, '[SSN_REMOVED]')
            removed_items.append(f"SSN: {ssn}")
        
        return cleaned_text, removed_items
    
    def get_text_statistics(self, text: str) -> Dict[str, any]:
        """Get comprehensive text statistics."""
        
        if not text:
            return {'error': 'Empty text provided'}
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Character statistics
        char_stats = {
            'total_chars': len(text),
            'chars_no_spaces': len(text.replace(' ', '')),
            'alphabetic_chars': len(re.findall(r'[a-zA-Z]', text)),
            'numeric_chars': len(re.findall(r'\d', text)),
            'whitespace_chars': len(re.findall(r'\s', text))
        }
        
        # Word statistics
        word_stats = {
            'total_words': len(words),
            'unique_words': len(set(word.lower() for word in words)),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'longest_word': max(words, key=len) if words else '',
            'shortest_word': min(words, key=len) if words else ''
        }
        
        # Sentence statistics
        sentence_stats = {
            'total_sentences': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_sentence_chars': sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        }
        
        return {
            'character_stats': char_stats,
            'word_stats': word_stats,
            'sentence_stats': sentence_stats,
            'readability_estimate': self._estimate_readability(word_stats, sentence_stats)
        }
    
    def _estimate_readability(self, word_stats: Dict, sentence_stats: Dict) -> str:
        """Simple readability estimation."""
        
        avg_word_length = word_stats.get('avg_word_length', 0)
        avg_sentence_length = sentence_stats.get('avg_sentence_length', 0)
        
        # Simple heuristic
        if avg_word_length < 4 and avg_sentence_length < 15:
            return 'Easy'
        elif avg_word_length < 6 and avg_sentence_length < 20:
            return 'Medium'
        else:
            return 'Hard'


# Example usage
if __name__ == "__main__":
    processor = TextProcessor()
    
    sample_text = """
    # AI Research Assistant
    
    This is a sample document for testing our AI Research Assistant.
    It contains multiple paragraphs and demonstrates text processing capabilities.
    
    ## Features
    
    - Document ingestion
    - Text processing  
    - Vector storage
    - Query capabilities
    
    Contact us at support@example.com or call 555-123-4567.
    """
    
    cleaned = processor.clean_text(sample_text)
    metadata = processor.extract_metadata_from_text(cleaned)
    sections = processor.extract_sections(cleaned)
    stats = processor.get_text_statistics(cleaned)
    
    print("Metadata:", metadata)
    print("Sections found:", len(sections))
    print("Statistics:", stats['word_stats'])