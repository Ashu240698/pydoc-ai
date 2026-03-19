"""
Central configuration module for PyDoc AI application.

This module defines all configuration parameters including file paths,
API keys, model selections, retrieval parameters, and UI settings.
All runtime settings should be defined here for easy maintenance and
environment-specific customization.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load environment variables from .env file for secure credential management
load_dotenv()


# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

# Base project directory (parent of this config file)
BASE_DIR = Path(__file__).parent

# Data storage directories for raw, processed, and embedding data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ============================================================================
# API & AUTHENTICATION
# ============================================================================

# Groq API key for LLM inference - loaded from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Sentence transformer model for generating dense vector embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cross-encoder model for ranking and re-ranking search results
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Large language model for response generation via Groq API
LLM_MODEL = "llama-3.1-8b-instant"

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================

# Size of text chunks for document splitting (characters)
CHUNK_SIZE = 1000

# Overlap between consecutive chunks to maintain context (characters)
CHUNK_OVERLAP = 150

# Number of top chunks to retrieve in initial hybrid search pass
HYBRID_SEARCH_TOP_K = 20

# Number of final chunks to return after re-ranking
FINAL_TOP_K = 4

# Weight for hybrid search: 0.5 = equal weight to BM25 and vector similarity
HYBRID_ALPHA = 0.5

# ============================================================================
# DOCUMENTATION SOURCES
# ============================================================================

# URLs for documentation sources to be collected and indexed
PYTHON_DOCS_URLS = {
    "stdlib": "https://docs.python.org/3/library/",
    "requests": "https://requests.readthedocs.io/en/latest/",
    "pandas": "https://pandas.pydata.org/docs/"
}


# ============================================================================
# USER INTERFACE SETTINGS
# ============================================================================

# Application title displayed in the UI
APP_TITLE = "🐍 PyDoc AI"

# Application subtitle displayed in the UI
APP_SUBTITLE = "Python Documentation Assistant"

# Maximum number of user-assistant exchanges to keep in conversation memory
MAX_CONVERSATION_HISTORY = 10
