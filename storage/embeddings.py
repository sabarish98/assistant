"""Embedding generation and management."""

from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import time

from core.config import config
from core.logger import app_logger


class EmbeddingManager:
    """Manages embedding generation using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       Default is a lightweight but effective model.
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._model_cache_dir = Path("./data/embedding_models")
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        app_logger.info(f"Initialized EmbeddingManager with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            app_logger.info(f"Loading sentence transformer model: {self.model_name}")
            start_time = time.time()
            
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self._model_cache_dir)
                )
                
                load_time = time.time() - start_time
                app_logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
                
            except Exception as e:
                app_logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
        
        return self._model
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if not texts:
            app_logger.warning("Empty text list provided for embedding generation")
            return []
        
        app_logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text.strip()]
            
            if len(valid_texts) != len(texts):
                app_logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
            
            if not valid_texts:
                return []
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Convert numpy arrays to lists for JSON serialization
            embedding_lists = embeddings.tolist()
            
            generation_time = time.time() - start_time
            app_logger.info(
                f"Generated {len(embedding_lists)} embeddings in {generation_time:.2f} seconds "
                f"({len(embedding_lists)/generation_time:.1f} embeddings/sec)"
            )
            
            return embedding_lists
            
        except Exception as e:
            app_logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            app_logger.warning("Empty text provided for single embedding generation")
            return []
        
        embeddings = self.generate_embeddings([text.strip()])
        return embeddings[0] if embeddings else []
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            app_logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: The query embedding to match against
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with 'index' and 'similarity' keys
        """
        if not query_embedding or not candidate_embeddings:
            return []
        
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sort by similarity descending and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        try:
            # Generate a test embedding to determine dimension
            test_embedding = self.generate_single_embedding("test")
            return len(test_embedding)
        except Exception as e:
            app_logger.error(f"Error getting embedding dimension: {e}")
            return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'model_cache_dir': str(self._model_cache_dir),
            'model_loaded': self._model is not None
        }
    
    def batch_similarity_search(
        self,
        query_embeddings: List[List[float]],
        document_embeddings: List[List[float]],
        threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            query_embeddings: List of query embeddings
            document_embeddings: List of document embeddings to search
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        results = []
        
        for query_embedding in query_embeddings:
            similarities = []
            
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = self.compute_similarity(query_embedding, doc_embedding)
                
                if similarity >= threshold:
                    similarities.append({
                        'index': i,
                        'similarity': similarity
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results.append(similarities)
        
        return results


# Global embedding manager instance
embedding_manager = EmbeddingManager()

__all__ = ["EmbeddingManager", "embedding_manager"]


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding manager
    manager = EmbeddingManager()
    
    sample_texts = [
        "Artificial intelligence is transforming the world",
        "Machine learning algorithms can process large datasets",
        "Natural language processing helps computers understand text",
        "The weather is sunny today",
        "I love eating pizza for dinner"
    ]
    
    print("Generating embeddings...")
    embeddings = manager.generate_embeddings(sample_texts, show_progress=True)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    
    # Test similarity
    if len(embeddings) >= 2:
        similarity = manager.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between first two texts: {similarity:.3f}")
    
    # Test similarity search
    query_embedding = embeddings[0]  # Use first text as query
    results = manager.find_most_similar(query_embedding, embeddings, top_k=3)
    
    print("\nTop 3 similar texts:")
    for result in results:
        idx = result['index']
        sim = result['similarity']
        print(f"  {idx}: {sample_texts[idx][:50]}... (similarity: {sim:.3f})")