"""
Interactive chat with PyDoc AI.
"""
from src.llm import LLMHandler

if __name__ == "__main__":
    handler = LLMHandler()
    handler.chat()
