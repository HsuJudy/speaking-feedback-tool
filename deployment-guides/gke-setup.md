# Google GKE Setup Guide for Speaking Feedback Tool

## Prerequisites
- Google Cloud SDK installed
- kubectl installed
- gcloud CLI configured

## 1. Install Required Tools

### Install Google Cloud SDK
```bash
# macOS
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Install kubectl
```bash
gcloud components install kubectl
```

## 2. Configure Google Cloud
```bash
# Initialize gcloud
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
```

## 3. Create GKE Cluster
```bash
# Create cluster with GPU support
gcloud container clusters create speaking-feedback-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --min-nodes 1 \
  --max-nodes 5 \
  --machine-type e2-medium \
  --enable-autoscaling \
  --enable-autorepair \
  --enable-autoupgrade

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster speaking-feedback-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --min-nodes 1 \
  --max-nodes 3 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling
```

## 4. Configure kubectl
```bash
gcloud container clusters get-credentials speaking-feedback-cluster --zone us-central1-a
```

## 5. Install GPU Drivers
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
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Apply your ingress
kubectl apply -f k8s/ingress.yaml
```

## 8. Configure SSL/TLS
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
- Use preemptible instances for non-critical workloads
- Enable cluster autoscaler
- Use committed use discounts
- Monitor resource usage with Cloud Monitoring

## Security Best Practices
- Enable Workload Identity
- Use Google Secret Manager
- Enable Cloud Audit Logs
- Use Google Cloud Armor 