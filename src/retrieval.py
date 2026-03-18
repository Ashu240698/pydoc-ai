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
        self.chunks = [item['chunk_text'] for item in data]
        for item in data:
            self.metadata.append({'chunk_id':item['chunk_id'], 'local_chunk_id':item['local_chunk_id'], 'source':item['title'], 'module':item['module'], 'url':item['url']})
        
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

def build_indexes():
    """Helper function to build indexes."""
    rag_retriever = RAGRetriever()
    rag_retriever.build_indexes()

if __name__ == "__main__":
    build_indexes()



