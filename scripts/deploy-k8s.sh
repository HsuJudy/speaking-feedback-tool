#!/bin/bash

# Kubernetes deployment script for Speaking Feedback Tool

set -e

NAMESPACE="speaking-feedback"

echo "☸️  Deploying to Kubernetes..."

# Create namespace
echo "📁 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply persistent volumes
echo "💾 Creating persistent volumes..."
kubectl apply -f k8s/persistent-volumes.yaml

# Apply ConfigMap and Secret
echo "🔧 Applying configuration..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy services
echo "🚀 Deploying services..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Apply ingress (if available)
if [ -f k8s/ingress.yaml ]; then
    echo "🌐 Applying ingress..."
    kubectl apply -f k8s/ingress.yaml
fi

echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/speaking-feedback-app -n $NAMESPACE

echo "🔍 Checking deployment status..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "✅ Kubernetes deployment complete!"
echo ""
echo "📊 Access your application:"
echo "  - kubectl port-forward svc/speaking-feedback-service 8080:80 -n $NAMESPACE"
echo "  - Then visit: http://localhost:8080"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: kubectl logs -f deployment/speaking-feedback-app -n $NAMESPACE"
echo "  - Scale up: kubectl scale deployment speaking-feedback-app --replicas=5 -n $NAMESPACE"
echo "  - Delete deployment: kubectl delete namespace $NAMESPACE" 