"""
Wrapper script to build FAISS and BM25 indexes.
Run this once after preprocessing to create indexes.
"""
from src.retrieval import build_indexes

if __name__ == "__main__":
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
    
    build_indexes()
    
    print("\n✅ Indexes are ready!")
    print("Next: Implement hybrid search and re-ranking\n")
