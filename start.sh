#!/bin/bash

# Quick Start Script for Base Knowledge Services

echo "🚀 Starting Base Knowledge Services..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Please create one with OPENAI_API_KEY"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d .venv ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || uv pip install -q -e .

# Initialize database
echo "🗄️  Initializing database..."
python scripts/init_db.py

# Start services in separate terminals (or use tmux/screen for background)
echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the services:"
echo ""
echo "Terminal 1 - Document Service (Port 8008):"
echo "  cd document-service && uvicorn main:app --reload --port 8008"
echo ""
echo "Terminal 2 - Chat Service (Port 8009):"
echo "  cd chat-service && uvicorn main:app --reload --port 8009"
echo ""
echo "Or use Docker Compose:"
echo "  docker-compose up -d"
echo ""
echo "Access:"
echo "  - Document API: http://localhost:8008/docs"
echo "  - Chat API: http://localhost:8009/docs"
echo "  - Chat UI: http://localhost:8009/chat"
