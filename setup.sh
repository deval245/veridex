#!/bin/bash

set -e

echo "🚀 Setting up VERIDEX..."

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - TMDB_API_KEY (get free key at: https://www.themoviedb.org/settings/api)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python examples/content_rating_demo.py"

