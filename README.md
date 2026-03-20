# 🐍 PyDoc AI - Python Documentation Assistant

AI-powered search assistant for Python documentation using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 **Hybrid Search**: BM25 keyword + FAISS semantic search
- 🎯 **Re-ranking**: Cross-encoder for better accuracy
- 💬 **Conversational**: Multi-turn conversations with memory
- 📚 **Source Citations**: Links to official documentation

## How to Use

1. **Get a Groq API Key** (free):
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up and create an API key
   - Free tier: 30 requests/minute

2. **Enter your API key** in the sidebar

3. **Ask Python questions!**
   - "How do I read a CSV file?"
   - "What is the requests library?"
   - "Compare pip and conda"

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Keyword Search**: BM25
- **Re-ranker**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: Groq (llama-3.1-8b-instant)

## Source Code

Full project on GitHub: [github.com/Ashu240698/pydoc-ai](https://github.com/Ashu240698/pydoc-ai)

---

**Note**: This app requires your own Groq API key for security. Your key is never stored and only used for your session.
