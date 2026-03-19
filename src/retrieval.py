"""
RAG Retrieval Module

This module implements a sophisticated Retrieval-Augmented Generation (RAG) retrieval system
that combines multiple search strategies for optimal document retrieval. The system features:

- Hybrid search: Combines BM25 keyword-based search with FAISS semantic vector search
- Cross-encoder re-ranking: Uses a transformer-based model to re-rank top candidates
- Efficient indexing: Pre-computed embeddings and indexes for fast retrieval
- Comprehensive logging: Tracks retrieval performance and query statistics

Key Components:
- RAGRetriever: Main class handling all retrieval operations
- Index building: Creates FAISS and BM25 indexes from document chunks
- Hybrid scoring: Weighted combination of keyword and semantic similarities
- Re-ranking: Refines top candidates using cross-encoder predictions

Dependencies:
- sentence-transformers: For embedding generation and re-ranking
- faiss: For efficient vector similarity search
- rank-bm25: For keyword-based BM25 scoring
- tqdm: For progress bars during processing
"""

import json
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import config
from src.logger import logger
import time


class RAGRetriever:
    """
    RAG Retrieval System with Hybrid Search and Re-ranking

    This class implements a state-of-the-art retrieval system that combines:
    1. BM25 keyword-based search for lexical matching
    2. FAISS vector similarity search for semantic matching
    3. Cross-encoder re-ranking for improved relevance

    The system is designed for production use with:
    - Pre-computed indexes for fast retrieval
    - Configurable search parameters
    - Comprehensive logging and performance tracking
    - Memory-efficient processing of large document collections

    Attributes:
        processed_dir (Path): Directory containing processed document chunks
        embeddings_dir (Path): Directory for storing/saving indexes and embeddings
        embedding_model (SentenceTransformer): Model for generating text embeddings
        reranker (CrossEncoder): Model for re-ranking candidate documents
        chunks (list): List of document chunk texts
        metadata (list): List of metadata dictionaries for each chunk
        faiss_index (faiss.Index): FAISS index for vector similarity search
        bm25 (BM25Okapi): BM25 index for keyword-based search
    """

    def __init__(self):
        """
        Initialize the RAG Retriever with models and directory setup.

        Sets up the retrieval system by:
        - Configuring data directories from config
        - Loading pre-trained embedding and re-ranking models
        - Initializing index placeholders

        Note: Indexes are loaded on-demand in retrieve() method.
        """
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.embeddings_dir = config.EMBEDDINGS_DIR
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        print("🔧 Initializing RAG Retriever...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

        self.chunks = None
        self.metadata = []
        self.faiss_index = None
        self.bm25 = None

        print("✅ Models loaded")

    def load_chunks(self):
        """
        Load processed document chunks and metadata from JSON file.

        Reads the pre-processed chunks from the configured processed data directory.
        Each chunk contains text content and associated metadata including source,
        module information, and URLs.

        The chunks are stored in self.chunks as a list of strings, and metadata
        is stored in self.metadata as a list of dictionaries.

        Raises:
            FileNotFoundError: If the chunks JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        chunks_file = self.processed_dir / "all_chunks.json"
        with open(chunks_file, "r", encoding="utf-8") as file:
            print(f"📂 Loading chunks from {chunks_file}...")
            data = json.load(file)

        # self.chunks = [chunk_data['text'] for chunk_data in data]
        self.chunks = [item['text'] for item in data]
        for item in data:
            self.metadata.append({'chunk_id':item['chunk_id'], 'local_chunk_id':item['local_chunk_id'], 'source':item['source'], 'module':item['module'], 'url':item['url']})

        print(f"✅ Loaded {len(self.chunks)} chunks")
    
    def build_embeddings(self):
        """
        Generate dense vector embeddings for all document chunks.

        Uses the configured SentenceTransformer model to encode each chunk into
        a high-dimensional vector representation. This enables semantic similarity
        search using FAISS.

        Returns:
            np.ndarray: 2D array of shape (num_chunks, embedding_dim) containing
                       normalized embeddings for all chunks.

        Note:
            This is a computationally intensive operation that should be run
            once during index building, not during query time.
        """
        print("🔢 Creating embeddings...")

        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)
        print(f"✅ Created embeddings: shape {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings):
        """
        Build FAISS index for efficient vector similarity search.

        Creates an L2 distance-based FAISS index from the provided embeddings.
        FAISS (Facebook AI Similarity Search) enables fast approximate nearest
        neighbor search in high-dimensional spaces.

        Args:
            embeddings (np.ndarray): 2D array of chunk embeddings

        Returns:
            faiss.IndexFlatL2: Trained FAISS index ready for similarity search

        Note:
            Uses exact L2 search for maximum accuracy. For very large datasets,
            consider approximate indexes like IVF or HNSW for better performance.
        """
        print("🗂️  Building FAISS index...")

        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
        return self.faiss_index
    
    def build_bm25_index(self):
        """
        Build BM25 index for keyword-based search.

        BM25 (Best Matching 25) is a probabilistic ranking function that estimates
        the relevance of documents to a search query. It improves upon TF-IDF by
        incorporating document length normalization and term frequency saturation.

        The index is built by tokenizing all chunks and creating a BM25 scorer.

        Returns:
            BM25Okapi: Trained BM25 index for keyword scoring

        Note:
            Tokenization is simple whitespace splitting with lowercasing.
            For more sophisticated tokenization, consider using NLTK or spaCy.
        """
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"✅ BM25 index built")
        return self.bm25
    
    def save_faiss_index(self, embeddings):
        """
        Persist all indexes and data to disk for fast loading.

        Saves the following components to the embeddings directory:
        - FAISS index: Vector similarity search index
        - Chunks: Original text chunks for retrieval
        - Metadata: Associated metadata for each chunk
        - BM25 index: Keyword-based search index

        Args:
            embeddings (np.ndarray): Chunk embeddings (used for validation)

        Note:
            Files are saved in pickle format for Python objects and FAISS
            native format for the vector index. This enables fast loading
            during inference.
        """
        print(f"💾 Saving indexes to {self.embeddings_dir}...")

        faiss_path = self.embeddings_dir / "chunks.faiss"
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"   ✅ Saved FAISS index")

        chunks_path = self.embeddings_dir / "chunks.pkl"
        with open(chunks_path, "wb") as chunk_file:
            pickle.dump(self.chunks, chunk_file)
        print(f"   ✅ Saved chunks")

        metadata_path = self.embeddings_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as meta_file:
            pickle.dump(self.metadata, meta_file)
        print(f"   ✅ Saved metadata")

        bm25_path = self.embeddings_dir / "bm25.pkl"
        with open(bm25_path, "wb") as bm25_file:
            pickle.dump(self.bm25, bm25_file)
        print(f"   ✅ Saved BM25 index")

    
    def load_indexes(self):
        """
        Load pre-built indexes and data from disk.

        Restores all persisted components from the embeddings directory:
        - FAISS index for vector search
        - Text chunks for result formatting
        - Metadata for result enrichment
        - BM25 index for keyword search

        This method enables fast startup by avoiding re-computation of
        embeddings and indexes during inference.

        Raises:
            FileNotFoundError: If any required index files are missing
            pickle.UnpicklingError: If index files are corrupted
        """
        print("📂 Loading indexes...")

        faiss_path = self.embeddings_dir / "chunks.faiss"
        self.faiss_index = faiss.read_index(str(faiss_path))
        print(f"   ✅ Loaded FAISS index ({self.faiss_index.ntotal} vectors)")

        chunks_path = self.embeddings_dir / "chunks.pkl"
        with open(chunks_path, "rb") as chunks_file:
            self.chunks = pickle.load(chunks_file)
        print(f"   ✅ Loaded {len(self.chunks)} chunks")

        metadata_path = self.embeddings_dir / "metadata.pkl"
        with open(metadata_path, "rb") as meta_file:
            self.metadata = pickle.load(meta_file)
        print(f"   ✅ Loaded metadata")

        bm25_path = self.embeddings_dir / "bm25.pkl"
        with open(bm25_path, "rb") as bm25_file:
            self.bm25 = pickle.load(bm25_file)
        print(f"   ✅ Loaded BM25 index")

    def hybrid_search(self, query, k=20, alpha=0.5):
        """
        Perform hybrid search combining BM25 keyword and FAISS semantic search.

        This method implements a sophisticated retrieval strategy that balances
        lexical matching (BM25) with semantic similarity (FAISS). The results
        are combined using a weighted scoring approach.

        Args:
            query (str): The search query string
            k (int): Number of top results to return (default: 20)
            alpha (float): Weight for BM25 scores (0.0 to 1.0). Higher alpha
                          emphasizes keyword matching, lower emphasizes semantic
                          similarity (default: 0.5)

        Returns:
            tuple: (top_indices, top_scores)
                - top_indices (list): Indices of top-k chunks in self.chunks
                - top_scores (list): Corresponding hybrid scores (0.0 to 1.0)

        Note:
            Scores from both systems are normalized to [0,1] before combination
            to ensure balanced contribution regardless of scale differences.
        """
        start_time = time.time()

        # Semantic search using FAISS
        query_embeddings = self.embedding_model.encode([query])
        faiss_dist, faiss_indices = self.faiss_index.search(query_embeddings, k=k)
        # Convert L2 distances to similarities (higher = more similar)
        faiss_similarities = 1 + (1 / faiss_dist[0])
        # Normalize to [0,1] range
        faiss_min = faiss_similarities.min()
        faiss_max = faiss_similarities.max()
        faiss_normalized = (faiss_similarities - faiss_min) / (faiss_max - faiss_min) if faiss_max > faiss_min else faiss_similarities

        # Keyword search using BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Normalize BM25 scores to [0,1]
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min) if bm25_max > bm25_min else bm25_scores

        # Combine scores with weighted average
        hybrid_scores = {}

        # Initialize with BM25 scores for all chunks
        for idx in range(len(self.chunks)):
            hybrid_scores[idx] = alpha * bm25_normalized[idx]

        # Add FAISS contributions for retrieved candidates
        for i, idx in enumerate(faiss_indices[0]):
            if idx in hybrid_scores:
                hybrid_scores[idx] += (1 - alpha) * faiss_normalized[i]
            else:
                hybrid_scores[idx] += (1 - alpha) * faiss_normalized[i]

        # Sort by combined score and return top-k
        sorted_results = sorted(hybrid_scores.items(), key=lambda k:k[1], reverse=True)
        top_indices = [idx for idx, score in sorted_results[:k]]
        top_scores = [score for idx, score in sorted_results[:k]]

        # Log performance metrics
        duration = time.time() - start_time
        logger.log_retrieval(
            query=query,
            stage='hybrid_search',
            num_candidates=len(top_indices),
            top_scores=top_scores
        )
        logger.log_performance('hybrid_search', duration)

        return top_indices, top_scores
    


    def rerank(self, query, candidate_indices, k=4):
        """
        Re-rank candidate documents using cross-encoder model.

        This method refines the initial retrieval results by using a more
        sophisticated cross-encoder model that considers query-document pairs
        jointly. Cross-encoders typically provide better relevance judgments
        than bi-encoder similarity scores.

        Args:
            query (str): The search query string
            candidate_indices (list): Indices of candidate chunks from hybrid search
            k (int): Number of top results to return after re-ranking (default: 4)

        Returns:
            tuple: (top_indices, top_scores)
                - top_indices (list): Indices of top-k re-ranked chunks
                - top_scores (list): Cross-encoder relevance scores

        Note:
            Cross-encoder predictions are computationally expensive, so this
            should only be applied to a small set of top candidates (typically
            10-50) rather than the entire corpus.
        """
        start_time = time.time()
        
        print(f"   Stage 2: Re-ranking {len(candidate_indices)} candidates...")

        # Prepare query-document pairs for cross-encoder
        candidate_chunks = [self.chunks[idx] for idx in candidate_indices]
        pairs = [[query, chunk] for chunk in candidate_chunks]

        # Get cross-encoder relevance scores
        re_ranked_scores = self.reranker.predict(pairs)

        # Sort by relevance score (descending)
        sorted_positions = np.argsort(re_ranked_scores)[::-1]
        final_top_4_positions = sorted_positions[:k]

        # Map back to original chunk indices
        final_top_4_indices = [candidate_indices[idx] for idx in final_top_4_positions]
        final_top_4_scores = [re_ranked_scores[idx] for idx in final_top_4_positions]

        # Log performance metrics
        duration = time.time() - start_time
        logger.log_retrieval(
            query=query,
            stage='rerank',
            num_candidates=len(final_top_4_indices),
            top_scores=final_top_4_scores
        )
        logger.log_performance('rerank', duration)

        return final_top_4_indices, final_top_4_scores
    

    def retrieve(self, query, k_hybrid=20, k_final=4, alpha=0.5):
        """
        Execute the complete retrieval pipeline: hybrid search + re-ranking.

        This is the main entry point for document retrieval. It orchestrates
        the two-stage process: first retrieving candidates using hybrid search,
        then re-ranking the top candidates with a cross-encoder model.

        Args:
            query (str): The search query string
            k_hybrid (int): Number of candidates from hybrid search (default: 20)
            k_final (int): Number of final results after re-ranking (default: 4)
            alpha (float): BM25 weight in hybrid search (default: 0.5)

        Returns:
            list: List of dictionaries containing retrieval results. Each dict has:
                - 'chunk_id': Index of the chunk in the corpus
                - 'text': The chunk text content
                - 'metadata': Associated metadata (source, module, url, etc.)
                - 'hybrid_score': Score from hybrid search (normalized)
                - 'rerank_score': Score from cross-encoder re-ranking

        Note:
            Indexes are loaded automatically if not already in memory.
            For production use, ensure indexes are pre-loaded for optimal performance.
        """
        # Load indexes if not loaded (FIX: Check all components)
        if self.chunks is None or self.faiss_index is None:
            self.load_indexes()

        print(f"\n🔍 Query: '{query}'")

        # self.build_indexes()
        
        # Stage 1: Hybrid search
        print(f"   Stage 1: Hybrid search (top {k_hybrid})...")
        hybrid_indices, hybrid_scores = self.hybrid_search(query, k=k_hybrid, alpha=alpha)

        # Stage 2: Re-ranking 
        top_indices, top_scores = self.rerank(query, hybrid_indices, k=k_final)

        results = []
        for i in range(min(k_final, len(top_indices))):
            idx = top_indices[i]
            results.append({
                'chunk_id':idx,
                'text':self.chunks[idx],
                'metadata':self.metadata[idx],
                'hybrid_score':hybrid_scores[hybrid_indices.index(idx)],
                'rerank_score':top_scores[i]
            })

        print(f"   ✅ Retrieved {len(results)} results")
        return results

        

    def build_indexes(self):
        """
        Build and persist all retrieval indexes from scratch.

        This method orchestrates the complete index construction pipeline:
        1. Load processed document chunks from disk
        2. Generate embeddings using the sentence transformer
        3. Build FAISS vector index for semantic search
        4. Build BM25 index for keyword search
        5. Save all components to disk for fast loading

        This is a one-time setup operation that should be run after data
        preprocessing. The resulting indexes enable fast retrieval during
        inference.

        Note:
            This operation can be time and memory intensive depending on
            corpus size. Monitor resource usage for large datasets.
        """
        self.load_chunks()
        embeddings = self.build_embeddings()
        faiss_index = self.build_faiss_index(embeddings)
        bm25_index = self.build_bm25_index()
        self.save_faiss_index(embeddings)

        print("\n" + "="*60)
        print("✅ INDEX BUILDING COMPLETE!")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   Total chunks: {len(self.chunks)}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   FAISS index size: {self.faiss_index.ntotal}")
        print("="*60 + "\n")
    

def build_indexes():
    """
    Standalone function to build retrieval indexes.

    This is a convenience function that can be called directly to construct
    all necessary indexes for the RAG system. It creates a new RAGRetriever
    instance and runs the complete index building pipeline.

    Usage:
        python -c "from src.retrieval import build_indexes; build_indexes()"

    Note:
        This function is primarily used for initial setup and testing.
        In production applications, consider using the RAGRetriever class directly.
    """
    rag_retriever = RAGRetriever()
    rag_retriever.build_indexes()


if __name__ == "__main__":
    # Entry point for command-line execution
    # Allows running index building as a standalone script
    build_indexes()