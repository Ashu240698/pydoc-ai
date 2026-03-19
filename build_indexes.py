"""
Wrapper script to build FAISS and BM25 indexes.

This module orchestrates the index building process, which creates
vector (FAISS) and keyword (BM25) search indexes from preprocessed
document chunks. Run this script once after preprocessing documents
to prepare the retrieval system for queries.

Usage:
    python build_indexes.py
"""
from src.retrieval import build_indexes


if __name__ == "__main__":
    # Display build process information
    print("\n" + "="*70)
    print("Building RAG Indexes")
    print("="*70)
    print("\nThis will:")
    print("  1. Load processed chunks")
    print("  2. Create embeddings (sentence-transformers)")
    print("  3. Build FAISS index (vector search)")
    print("  4. Build BM25 index (keyword search)")
    print("  5. Save all indexes to disk")
    print("\nThis may take 1-2 minutes...\n")

    # Trigger the index building process
    build_indexes()

    # Confirm successful completion
    print("\n✅ Indexes are ready!")
    print("Next: Implement hybrid search and re-ranking\n")
