#!/bin/bash

# Cloud Kubernetes Setup Helper for Speaking Feedback Tool

echo "☁️  CLOUD KUBERNETES SETUP HELPER"
echo "=================================="
echo ""
echo "Choose your cloud provider:"
echo "1. AWS EKS (Enterprise, GPU support, higher cost)"
echo "2. Google GKE (Good GPU support, moderate cost)"
echo "3. DigitalOcean (Simple, no GPU, lower cost)"
echo "4. Local development (Docker Compose)"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Setting up AWS EKS..."
        echo ""
        echo "📋 Prerequisites:"
        echo "  - AWS account with billing enabled"
        echo "  - AWS CLI configured"
        echo "  - kubectl installed"
        echo "  - eksctl installed"
        echo ""
        echo "💰 Estimated cost: $200-500/month"
        echo "🎯 Best for: Production, GPU workloads, enterprise"
        echo ""
        echo "📖 Follow the guide: deployment-guides/aws-eks-setup.md"
        echo ""
        echo "🔧 Quick setup commands:"
        echo "  brew install awscli kubectl eksctl"
        echo "  aws configure"
        echo "  eksctl create cluster --name speaking-feedback-cluster --region us-west-2"
        ;;
    2)
        echo "🚀 Setting up Google GKE..."
        echo ""
        echo "📋 Prerequisites:"
        echo "  - Google Cloud account"
        echo "  - Google Cloud SDK installed"
        echo "  - kubectl installed"
        echo ""
        echo "💰 Estimated cost: $150-400/month"
        echo "🎯 Best for: ML workloads, good GPU support"
        echo ""
        echo "📖 Follow the guide: deployment-guides/gke-setup.md"
        echo ""
        echo "🔧 Quick setup commands:"
        echo "  curl https://sdk.cloud.google.com | bash"
        echo "  gcloud init"
        echo "  gcloud container clusters create speaking-feedback-cluster"
        ;;
    3)
        echo "🚀 Setting up DigitalOcean Kubernetes..."
        echo ""
        echo "📋 Prerequisites:"
        echo "  - DigitalOcean account"
        echo "  - doctl CLI installed"
        echo "  - kubectl installed"
        echo ""
        echo "💰 Estimated cost: $50-150/month"
        echo "🎯 Best for: Development, simple setup, lower cost"
        echo "⚠️  Note: No GPU support available"
        echo ""
        echo "📖 Follow the guide: deployment-guides/digitalocean-k8s-setup.md"
        echo ""
        echo "🔧 Quick setup commands:"
        echo "  brew install doctl kubectl"
        echo "  doctl auth init"
        echo "  doctl kubernetes cluster create speaking-feedback-cluster"
        ;;
    4)
        echo "🚀 Setting up local development..."
        echo ""
        echo "📋 Prerequisites:"
        echo "  - Docker Desktop installed"
        echo "  - 8GB+ RAM available"
        echo ""
        echo "💰 Cost: Free (local resources only)"
        echo "🎯 Best for: Development, testing, learning"
        echo ""
        echo "🔧 Quick setup commands:"
        echo "  ./scripts/deploy-docker.sh"
        echo "  ./scripts/setup-monitoring.sh"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📋 Next steps:"
echo "1. Follow the guide for your chosen provider"
echo "2. Set up your environment variables"
echo "3. Deploy your application"
echo "4. Set up monitoring"
echo ""
echo "🎯 Your Speaking Feedback Tool will be ready for production!" 