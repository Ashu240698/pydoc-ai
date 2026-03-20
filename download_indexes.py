"""Download indexes from HuggingFace dataset if not present."""
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_indexes_if_needed():
    """Download indexes from HF dataset on first run."""
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    files = ["bm25.pkl", "chunks.faiss", "chunks.pkl", "metadata.pkl"]
    
    for filename in files:
        local_path = embeddings_dir / filename
        if not local_path.exists():
            print(f"📥 Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id="ashu240698/pydoc-ai-indexes",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(embeddings_dir),
                    local_dir_use_symlinks=False
                )
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                raise
        else:
            print(f"✅ {filename} already exists")

if __name__ == "__main__":
    download_indexes_if_needed()