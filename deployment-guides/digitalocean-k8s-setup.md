# DigitalOcean Kubernetes Setup Guide for Speaking Feedback Tool

## Prerequisites
- DigitalOcean account
- doctl CLI installed
- kubectl installed

## 1. Install Required Tools

### Install doctl (DigitalOcean CLI)
```bash
# macOS
brew install doctl

# Linux
snap install doctl
```

### Install kubectl
```bash
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## 2. Configure DigitalOcean
```bash
# Authenticate with DigitalOcean
doctl auth init

# List your projects
doctl projects list
```

## 3. Create Kubernetes Cluster
```bash
# Create cluster (note: DigitalOcean doesn't support GPU nodes)
doctl kubernetes cluster create speaking-feedback-cluster \
  --region nyc1 \
  --size s-2vcpu-4gb \
  --count 3 \
  --wait

# Get cluster credentials
doctl kubernetes cluster kubeconfig save speaking-feedback-cluster
```

## 4. Deploy Your Application
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volumes.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 5. Set Up Load Balancer
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/do/deploy.yaml

# Apply your ingress
kubectl apply -f k8s/ingress.yaml
```

## 6. Configure SSL/TLS
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## 7. Monitor Your Deployment
```bash
# Check pod status
kubectl get pods -n speaking-feedback

# Check services
kubectl get services -n speaking-feedback

# Check ingress
kubectl get ingress -n speaking-feedback
```

## 8. Access Your Application
```bash
# Get the Load Balancer URL
kubectl get service speaking-feedback-service -n speaking-feedback
```

## Cost Optimization
- Use smaller node sizes for development
- Enable cluster autoscaler
- Monitor resource usage
- Use DigitalOcean's monitoring tools

## Security Best Practices
- Use DigitalOcean's firewall
- Enable monitoring and alerting
- Use secrets management
- Regular security updates 