#!/bin/bash

echo "🚀 Setting up PyDoc AI..."

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this from the project root directory"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ../rag_venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed data/embeddings logs

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Please create .env and add your GROQ_API_KEY"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your GROQ_API_KEY to .env file"
echo "2. Run: python src/data_collection.py"
echo "3. Then: streamlit run app.py"
