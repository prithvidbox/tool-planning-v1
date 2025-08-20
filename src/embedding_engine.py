"""
Embedding generation and FAISS indexing for intent matching.
"""

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from .config_parser import ConfigParser, IntentConfig


class EmbeddingEngine:
    """Handles embedding generation and vector similarity search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine with performance optimizations.
        
        Args:
            model_name: HuggingFace model name for sentence transformers
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        # Load model with optimizations for faster inference
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # Set to evaluation mode for faster inference
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.intent_metadata: List[Dict[str, Any]] = []
        self.intents: List[IntentConfig] = []
        
        # Performance optimization: cache for repeated queries
        self.query_cache = {}
        self.cache_size_limit = 100
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
        
    def build_index(self, config_parser: ConfigParser) -> None:
        """Build FAISS index from intent configurations."""
        logger.info("Building FAISS index from intent configurations")
        
        # Store intents for later retrieval
        self.intents = list(config_parser.intents)
        
        # Generate embeddings for all intents
        intent_texts = config_parser.get_all_intent_texts()
        embeddings = self.generate_embeddings(intent_texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata
        self.intent_metadata = [
            {
                'intent_name': intent.intent,
                'platform': intent.platform,
                'description': intent.description,
                'example_count': len(intent.examples),
                'tool_plan_count': len(intent.tool_plan)
            }
            for intent in self.intents
        ]
        
        logger.info(f"Built FAISS index with {self.index.ntotal} intents")
        
    def search_similar_intents(self, query: str, top_k: int = 5) -> List[Tuple[IntentConfig, float]]:
        """
        Search for similar intents using semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of top results to return
            
        Returns:
            List of (IntentConfig, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        # Generate query embedding
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Return results with intent configs and scores
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.intents):  # Valid index
                intent_config = self.intents[idx]
                results.append((intent_config, float(similarity)))
                
        logger.debug(f"Found {len(results)} similar intents for query: '{query}'")
        return results
        
    def find_best_intent(self, query: str, confidence_threshold: float = 0.8) -> Optional[Tuple[IntentConfig, float]]:
        """
        Find the best matching intent above confidence threshold.
        
        Args:
            query: User query text
            confidence_threshold: Minimum similarity score required
            
        Returns:
            (IntentConfig, similarity_score) or None if no match above threshold
        """
        results = self.search_similar_intents(query, top_k=1)
        
        if results and results[0][1] >= confidence_threshold:
            intent, score = results[0]
            logger.info(f"Matched intent '{intent.intent}' (platform: {intent.platform}) with confidence {score:.3f}")
            return (intent, score)
        else:
            best_score = results[0][1] if results else 0.0
            logger.warning(f"No intent matched above threshold {confidence_threshold}. Best score: {best_score:.3f}")
            return None
            
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
            
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save metadata and intents
        metadata = {
            'intent_metadata': self.intent_metadata,
            'intents': self.intents,
            'embedding_dim': self.embedding_dim
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved metadata to {metadata_path}")
        
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load FAISS index and metadata from disk."""
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.intent_metadata = metadata['intent_metadata']
        self.intents = metadata['intents']
        self.embedding_dim = metadata['embedding_dim']
        
        logger.info(f"Loaded index with {self.index.ntotal} intents")
        
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {'status': 'not_built'}
            
        platform_counts = {}
        for intent in self.intents:
            platform_counts[intent.platform] = platform_counts.get(intent.platform, 0) + 1
            
        return {
            'status': 'built',
            'total_intents': len(self.intents),
            'embedding_dimension': self.embedding_dim,
            'platform_breakdown': platform_counts,
            'model_name': self.model._modules['0'].auto_model.name_or_path
        }


class RRFEngine:
    """Reciprocal Rank Fusion for combining multiple ranking sources."""
    
    @staticmethod
    def combine_rankings(rankings: List[List[Tuple[Any, float]]], k: int = 60) -> List[Tuple[Any, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            rankings: List of ranked results, each as list of (item, score) tuples
            k: RRF parameter (typically 60)
            
        Returns:
            Combined ranked list of (item, rrf_score) tuples
        """
        rrf_scores = {}
        
        # Calculate RRF scores
        for ranking in rankings:
            for rank, (item, _) in enumerate(ranking):
                item_key = f"{item.intent}_{item.platform}"
                if item_key not in rrf_scores:
                    rrf_scores[item_key] = {'item': item, 'score': 0.0}
                rrf_scores[item_key]['score'] += 1.0 / (k + rank + 1)
                
        # Sort by RRF score and return
        combined = [(data['item'], data['score']) for data in rrf_scores.values()]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
