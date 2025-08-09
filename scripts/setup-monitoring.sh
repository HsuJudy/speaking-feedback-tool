#!/bin/bash

# Monitoring setup script for Speaking Feedback Tool

set -e

echo "📊 Setting up Prometheus and Grafana monitoring..."

# Create monitoring directories
mkdir -p monitoring/rules
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards

echo "✅ Monitoring directories created"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start monitoring stack
echo "🚀 Starting monitoring stack with Docker Compose..."
docker-compose up -d prometheus grafana

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking monitoring services..."
docker-compose ps prometheus grafana

echo ""
echo "✅ Monitoring setup complete!"
echo ""
echo "📊 Access your monitoring dashboards:"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "📋 Next steps:"
echo "  1. Open Grafana at http://localhost:3000"
echo "  2. Login with admin/admin"
echo "  3. Add Prometheus data source: http://prometheus:9090"
echo "  4. Import the Speaking Feedback dashboard"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f prometheus"
echo "  - Restart: docker-compose restart prometheus grafana"
echo "  - Stop: docker-compose stop prometheus grafana" 