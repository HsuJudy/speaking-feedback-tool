#!/bin/bash

# Docker deployment script for Speaking Feedback Tool

set -e

echo "🐳 Building Docker image..."
docker build -t speaking-feedback:latest .

echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Checking service status..."
docker-compose ps

echo "✅ Deployment complete!"
echo "📊 Services available:"
echo "  - App: http://localhost:5001"
echo "  - MLflow: http://localhost:5000"
echo "  - Triton: http://localhost:8000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"

echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart: docker-compose restart" 