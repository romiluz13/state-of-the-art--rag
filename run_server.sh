#!/bin/bash
# Start SOTA RAG API server

set -e

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example and configure it."
    exit 1
fi

# Start server
echo "Starting SOTA RAG API server..."
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
