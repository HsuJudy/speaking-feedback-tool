# AWS EKS Setup Guide for Speaking Feedback Tool

## Prerequisites
- AWS CLI installed and configured
- kubectl installed
- eksctl installed

## 1. Install Required Tools

### Install AWS CLI
```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Install kubectl
```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### Install eksctl
```bash
# macOS
brew install eksctl

# Linux
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

## 2. Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-west-2)
# Enter your output format (json)
```

## 3. Create EKS Cluster
```bash
# Create cluster with GPU support
eksctl create cluster \
  --name speaking-feedback-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# For GPU workloads, create additional node group
eksctl create nodegroup \
  --cluster speaking-feedback-cluster \
  --region us-west-2 \
  --name gpu-workers \
  --node-type g4dn.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

## 4. Configure kubectl
```bash
aws eks update-kubeconfig --region us-west-2 --name speaking-feedback-cluster
```

## 5. Install GPU Operators
```bash
# Install NVIDIA GPU operator
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
```

## 6. Deploy Your Application
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volumes.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 7. Set Up Load Balancer
```bash
# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=speaking-feedback-cluster
```

## 8. Configure Ingress
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/aws/deploy.yaml

# Apply your ingress
kubectl apply -f k8s/ingress.yaml
```

## 9. Monitor Your Deployment
```bash
# Check pod status
kubectl get pods -n speaking-feedback

# Check services
kubectl get services -n speaking-feedback

# Check ingress
kubectl get ingress -n speaking-feedback
```

## 10. Access Your Application
```bash
# Get the Load Balancer URL
kubectl get service speaking-feedback-service -n speaking-feedback
```

## Cost Optimization
- Use Spot instances for non-critical workloads
- Set up auto-scaling based on demand
- Use AWS Graviton instances for cost savings
- Enable cluster autoscaler

## Security Best Practices
- Enable AWS IAM roles for service accounts
- Use AWS Secrets Manager for sensitive data
- Enable AWS CloudTrail for audit logging
- Use AWS WAF for web application firewall 