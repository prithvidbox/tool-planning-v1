"""
OpenAI-based embedding generation and FAISS indexing for intent matching.
"""

import numpy as np
import faiss
import pickle
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
from .config_parser import ConfigParser, IntentConfig

# Load environment variables
load_dotenv()


class OpenAIEmbeddingEngine:
    """Handles OpenAI embedding generation and vector similarity search."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding engine with OpenAI embeddings.
        
        Args:
            model_name: OpenAI embedding model name
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = os.getenv('OPENAI_EMBEDDING_MODEL', model_name)
        
        # Set embedding dimension based on model
        if "text-embedding-3-small" in self.model_name:
            self.embedding_dim = 1536
        elif "text-embedding-3-large" in self.model_name:
            self.embedding_dim = 3072
        elif "text-embedding-ada-002" in self.model_name:
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 1536  # Default
            
        logger.info(f"Initialized OpenAI embedding engine with model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.intent_metadata: List[Dict[str, Any]] = []
        self.intents: List[IntentConfig] = []
        
        # Performance optimization: cache for repeated queries
        self.query_cache = {}
        self.cache_size_limit = 100
        
        # Performance tracking
        self.last_token_usage = {}
        self.last_timing = {}
        self.total_tokens_used = 0
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 2048) -> np.ndarray:
        """Generate embeddings for a list of texts using OpenAI API with optimized batching."""
        logger.info(f"Generating OpenAI embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        # Process in larger batches (OpenAI allows up to 8192 input items)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                # Reduce batch size and retry on error
                if batch_size > 100:
                    logger.info(f"Retrying with smaller batch size: {batch_size // 2}")
                    return self.generate_embeddings(texts, batch_size // 2)
                raise
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        return embeddings
        
    def build_index(self, config_parser: ConfigParser) -> None:
        """Build FAISS index from intent configurations using OpenAI embeddings."""
        logger.info("Building FAISS index from intent configurations using OpenAI")
        
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
        
        logger.info(f"Built FAISS index with {self.index.ntotal} intents using OpenAI embeddings")
        
    def search_similar_intents(self, query: str, top_k: int = 5) -> List[Tuple[IntentConfig, float]]:
        """
        Search for similar intents using OpenAI embeddings and semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of top results to return
            
        Returns:
            List of (IntentConfig, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Check cache first
        if query in self.query_cache:
            logger.debug(f"Using cached embedding for query: '{query}'")
            query_embedding = self.query_cache[query]
            embedding_time = 0.0  # No API call needed
            embedding_tokens = 0
        else:
            # Track timing for embedding generation
            start_time = time.time()
            
            # Generate query embedding using OpenAI
            try:
                response = self.client.embeddings.create(
                    input=[query],
                    model=self.model_name
                )
                
                query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
                
                # Track performance
                end_time = time.time()
                embedding_time = end_time - start_time
                
                if hasattr(response, 'usage') and response.usage:
                    embedding_tokens = response.usage.total_tokens
                    self.total_tokens_used += embedding_tokens
                else:
                    embedding_tokens = 0
                
                # Cache the embedding
                if len(self.query_cache) < self.cache_size_limit:
                    self.query_cache[query] = query_embedding
                    
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                raise
            
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        search_start = time.time()
        similarities, indices = self.index.search(query_embedding, top_k)
        search_time = time.time() - search_start
        
        # Update timing info
        self.last_timing = {
            'embedding_time': embedding_time,
            'search_time': search_time,
            'total_time': embedding_time + search_time,
            'timestamp': time.time()
        }
        
        # Update token usage
        self.last_token_usage = {
            'embedding_tokens': embedding_tokens,
            'operation': 'intent_search',
            'cached': query in self.query_cache and embedding_time == 0.0
        }
        
        # Return results with intent configs and scores
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.intents):  # Valid index
                intent_config = self.intents[idx]
                results.append((intent_config, float(similarity)))
                
        logger.debug(f"Found {len(results)} similar intents for query: '{query}' in {search_time:.3f}s")
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
            
        logger.info(f"Saving OpenAI-based index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save metadata and intents
        metadata = {
            'intent_metadata': self.intent_metadata,
            'intents': self.intents,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'engine_type': 'openai'
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved OpenAI metadata to {metadata_path}")
        
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load FAISS index and metadata from disk."""
        logger.info(f"Loading OpenAI-based index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.intent_metadata = metadata['intent_metadata']
        self.intents = metadata['intents']
        self.embedding_dim = metadata['embedding_dim']
        
        # Ensure we're using the same model as when saved
        if 'model_name' in metadata:
            self.model_name = metadata['model_name']
            
        logger.info(f"Loaded OpenAI-based index with {self.index.ntotal} intents")
        
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
            'model_name': self.model_name,
            'engine_type': 'openai'
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
