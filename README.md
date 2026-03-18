# 🐍 PyDoc AI - Python Documentation Assistant

AI-powered search assistant for Python documentation using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 **Hybrid Search**: Combines keyword (BM25) and semantic (FAISS) search
- 🎯 **Re-ranking**: Uses cross-encoder for accurate result ordering
- 💬 **Conversation History**: Maintains context across questions
- 📚 **Source Citations**: Links back to official documentation
- ⚡ **Fast**: Responses in under 3 seconds

## Supported Libraries

- Python Standard Library
- requests
- pandas

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Keyword Search**: BM25
- **Re-ranker**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: Groq (llama-3.1-8b-instant)
- **UI**: Streamlit

## Installation
```bash
# Clone repository
git clone <your-repo>
cd pydoc-ai

# Run setup script
./setup.sh

# Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" > .env
```

## Usage

### 1. Collect Documentation Data
```bash
python src/data_collection.py
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Ask Questions!
```
User: "How do I read a CSV file?"
AI: "Use the csv module. Here's how: ..."
```

## Project Structure
```
pydoc-ai/
├── data/              # Documentation data
├── src/               # Source code
├── app.py            # Streamlit UI
├── config.py         # Configuration
└── requirements.txt  # Dependencies
```

## Development

Built over 2 weeks as part of RAG learning curriculum.

## License

MIT
