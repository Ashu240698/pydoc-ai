"""
Wrapper script to execute document preprocessing from project root.

This module provides a convenient entry point for running the document
preprocessing pipeline. It loads raw documentation files, cleans and
chunks them for later indexing and retrieval.

Usage:
    python preprocess_docs.py

Note: Run this before build_indexes.py in the setup workflow.
"""
from src.preprocessing import main


if __name__ == "__main__":
    # Execute the main preprocessing pipeline
    main()
