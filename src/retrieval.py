"""
RAG Retrieval Module
Implements hybrid search with BM25 and FAISS, plus cross-encoder re-ranking.
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

class RAGRetriever:
    """
    RAG retrieval system with hybrid search and re-ranking.
    """

    def __init__(self):
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
        """Load processed chunks from JSON."""

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
        """Create embeddings for all chunks."""
        print("🔢 Creating embeddings...")

        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)
        print(f"✅ Created embeddings: shape {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings):
        """Build FAISS index from embeddings."""
        print("🗂️  Building FAISS index...")

        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
        return self.faiss_index
    
    def build_bm25_index(self):

        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"✅ BM25 index built")
        return self.bm25
    
    def save_faiss_index(self, embeddings):
        """Save FAISS index, BM25, and metadata to disk."""
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
        """Load pre-built indexes from disk."""
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
        Hybrid search combining BM25 (keyword) and FAISS (semantic).
        
        Args:
            query: Search query string
            k: Number of results to return
            alpha: Weight for BM25 (1-alpha for FAISS)
        
        Returns:
            top_indices: List of chunk indices
            top_scores: List of combined scores
        """

        query_embeddings = self.embedding_model.encode([query])
        faiss_dist, faiss_indices = self.faiss_index.search(query_embeddings, k=k)
        faiss_similarities = 1 + (1 / faiss_dist[0])
        faiss_min = faiss_similarities.min()
        faiss_max = faiss_similarities.max()
        faiss_normalized = (faiss_similarities - faiss_min) / (faiss_max - faiss_min) if faiss_max > faiss_min else faiss_similarities

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min) if bm25_max > bm25_min else bm25_scores

        hybrid_scores = {}

        for idx in range(len(self.chunks)):
            hybrid_scores[idx] = alpha * bm25_normalized[idx]

        for i, idx in enumerate(faiss_indices[0]):
            if idx in hybrid_scores:
                hybrid_scores[idx] += (1 - alpha) * faiss_normalized[i]
            else:
                hybrid_scores[idx] += (1 - alpha) * faiss_normalized[i]

        sorted_results = sorted(hybrid_scores.items(), key=lambda k:k[1], reverse=True)
        top_indices = [idx for idx, score in sorted_results[:k]]
        top_scores = [score for idx, score in sorted_results[:k]]

        return top_indices, top_scores
    



    def retrieve(self, query, k_hybrid=20, k_final=4, alpha=0.5):
        """
        Full retrieval pipeline: hybrid search + re-ranking.
        
        Args:
            query: Search query
            k_hybrid: Number of candidates from hybrid search
            k_final: Number of final results after re-ranking
            alpha: BM25 weight in hybrid search
        
        Returns:
            results: List of dicts with chunk text, metadata, and scores
        """

        if self.chunks is None:
            self.load_chunks()

        print(f"\n🔍 Query: '{query}'")
        
        # Stage 1: Hybrid search
        print(f"   Stage 1: Hybrid search (top {k_hybrid})...")
        top_indices, top_scores = self.hybrid_search(query, k_hybrid, alpha)

        results = []
        for i in range(min(k_final, len(top_indices))):
            idx = top_indices[i]
            results.append({
                'chunk_id':idx,
                'text':self.chunks[idx],
                'metadata':self.metadata[idx],
                'score':top_scores[i]
            })

        print(f"   ✅ Retrieved {len(results)} results")
        return results

        

    def build_indexes(self):
        """
        Main function to build all indexes.
        Run this once to create FAISS and BM25 indexes.
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

        query_text = "How do I read a CSV file?"
        results = self.retrieve(query_text, k_final=4)

        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                print(f"\n[{i}] Score: {result['score']:.3f}")
                print(f"    Source: {result['metadata']['source']}")
                print(f"    Module: {result['metadata'].get('module', 'N/A')}")
                print(f"    Text: {result['text'][:150]}...")
            else:
                print(f"Warning: result is a {type(result)}, not a dict.")
    

def build_indexes():
    """Helper function to build indexes."""
    rag_retriever = RAGRetriever()
    rag_retriever.build_indexes()

if __name__ == "__main__":
    build_indexes()