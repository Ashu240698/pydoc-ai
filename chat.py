"""
Interactive chat interface for PyDoc AI.

This module provides a command-line interface for users to interact
with the PyDoc AI assistant. It initializes the LLM handler and
starts an interactive chat session for real-time Q&A about Python
documentation.

Usage:
    python chat.py
"""
from src.llm import LLMHandler


if __name__ == "__main__":
    # Initialize the LLM handler with retrieval and conversation capabilities
    handler = LLMHandler()

    # Start the interactive chat loop
    handler.chat()
