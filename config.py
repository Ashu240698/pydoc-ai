"""
Configuration settings for PyDoc AI
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
HYBRID_SEARCH_TOP_K = 20
FINAL_TOP_K = 4
HYBRID_ALPHA = 0.5  # 0.5 = equal weight to BM25 and vector search

# Documentation sources
PYTHON_DOCS_URLS = {
    "stdlib": "https://docs.python.org/3/library/",
    "requests": "https://requests.readthedocs.io/en/latest/",
    "pandas": "https://pandas.pydata.org/docs/"
}

# UI settings
APP_TITLE = "🐍 PyDoc AI"
APP_SUBTITLE = "Python Documentation Assistant"
MAX_CONVERSATION_HISTORY = 10  # Keep last 10 exchanges
