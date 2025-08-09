# 🎤 Speaking Feedback Tool

A real-time speaking feedback and sentiment analysis tool that integrates with Zoom meetings to provide instant feedback on speaking quality, tone, and sentiment. Features a modern web interface, complete CI/CD pipeline, and enterprise-grade MLOps stack.

## 🚀 Features

### **🎯 Core Functionality**
- **Real-time Zoom Integration**: Automatically processes Zoom meetings via webhooks
- **Sentiment Analysis**: Analyzes speaking tone and sentiment using ML models
- **Audio Processing**: Extracts and processes audio from Zoom recordings
- **Stress Level Detection**: Identifies speaking anxiety and stress patterns
- **Emotion Recognition**: Detects emotional states during presentations

### **🎨 Modern Web Interface**
- **Responsive Dashboard**: Real-time metrics, charts, and insights
- **Meeting Management**: Search, filter, and analyze meeting history
- **Advanced Analytics**: Interactive charts and trend analysis
- **Settings Management**: Secure configuration with environment variables
- **Mobile-Friendly**: Works perfectly on all devices

### **🏗️ Enterprise MLOps Stack**
- **DVC**: Data versioning and pipeline management
- **Weights & Biases**: Experiment tracking and model management
- **MLflow**: Model serving and lifecycle management
- **NVIDIA Triton**: GPU-accelerated model serving
- **NeMo Models**: State-of-the-art speech AI models

### **🔧 DevOps & CI/CD**
- **GitHub Actions**: Complete CI/CD pipeline automation
- **Docker & Kubernetes**: Production-ready containerization
- **Prometheus & Grafana**: Real-time monitoring and alerting
- **Security Scanning**: Automated vulnerability detection
- **Multi-Environment**: Staging and production deployments

### **🔒 Security & Compliance**
- **Local Processing**: All analysis happens on your server
- **Secure Webhooks**: HMAC signature verification
- **Environment Variables**: Credentials stored securely
- **Data Privacy**: Automatic cleanup of sensitive data

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Zoom Meeting  │───▶│  Flask Webhook  │───▶│  ML Pipeline    │
│                 │    │   Endpoint      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  SQLite DB      │    │  Analysis       │
                       │  (Results)      │    │  Results        │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Modern Web     │    │  Real-time      │
                       │  Interface      │    │  Dashboard      │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Zoom Developer Account
- ngrok (for local development)
- ffmpeg (for audio processing)
- Docker (for containerized deployment)
- Kubernetes cluster (for production deployment)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HsuJudy/speaking-feedback-tool.git
cd speaking-feedback-tool/vibe-check
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp env_example.txt .env
# Edit .env with your Zoom credentials (see ENV_SETUP_GUIDE.md)
```

### 4. Install ffmpeg
**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🔧 Configuration

### Zoom App Setup

1. **Create Zoom App** at [Zoom App Marketplace](https://marketplace.zoom.us/)
2. **Choose "General Meeting App"** (not "Webhook only")
3. **Configure App Settings:**

#### Meeting Tab:
- **Webhook URL:** `https://your-domain.com/webhook/zoom`
- **Events:** `meeting.started`, `meeting.ended`, `recording.completed`

#### App Credentials Tab:
- Copy **Client ID** and **Client Secret**

#### Account Tab:
- Copy **Account ID**

#### OAuth Tab:
- **Redirect URL:** `https://your-domain.com/oauth/callback`

### Environment Variables

Create a `.env` file with your Zoom credentials:

```bash
# Zoom API Credentials
ZOOM_CLIENT_ID=your_client_id
ZOOM_CLIENT_SECRET=your_client_secret
ZOOM_ACCOUNT_ID=your_account_id
ZOOM_WEBHOOK_SECRET=your_webhook_secret

# Application Settings
FLASK_ENV=development
PORT=5001
RECORDINGS_DIR=zoom_recordings

# Optional: MLOps Stack
WANDB_API_KEY=your_wandb_api_key
MLFLOW_TRACKING_URI=http://localhost:5000
```

**📖 See [ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md) for detailed setup instructions.**

## 🚀 Usage

### Local Development

1. **Start the Flask App:**
```bash
python app.py
```

2. **Access the Web Interface:**
```
http://localhost:5001
```

3. **Start ngrok (for webhook testing):**
```bash
ngrok http 5001
```

4. **Update Zoom webhook URL** with your ngrok URL:
```
https://your-ngrok-url.ngrok-free.app/webhook/zoom
```

### Production Deployment

#### Option 1: Docker & Kubernetes (Recommended)
```bash
# Deploy with Docker Compose (local)
./scripts/deploy-docker.sh

# Deploy to Kubernetes
./scripts/deploy-k8s.sh

# Set up cloud Kubernetes
./scripts/setup-cloud-k8s.sh
```

#### Option 2: Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
heroku config:set ZOOM_CLIENT_ID=your_client_id
# ... set other environment variables
```

#### Option 3: Railway
```bash
# Connect your GitHub repo to Railway
# Set environment variables in Railway dashboard
```

#### Option 4: DigitalOcean App Platform
```bash
# Connect your GitHub repo to DigitalOcean
# Set environment variables in App Platform dashboard
```

## 📊 Web Interface

### **Dashboard** (`/`)
- Real-time metrics and system status
- Interactive charts (sentiment trends, emotion distribution)
- Recent meetings with analysis results
- System alerts and notifications

### **Meetings** (`/meetings`)
- Search and filter meeting history
- Detailed meeting information
- Download analysis results
- Meeting analytics and insights

### **Analytics** (`/analytics`)
- Advanced trend analysis
- Performance metrics
- Correlation analysis
- Export functionality

### **Settings** (`/settings`)
- Configuration management
- Environment variable status
- Test connections (Zoom, ML models)
- Database backup and management

## 🔧 API Endpoints

### Webhook Endpoints
- `POST /webhook/zoom` - Zoom webhook receiver
- `GET /oauth/callback` - OAuth callback handler

### Dashboard Endpoints
- `GET /` - Main dashboard
- `GET /meetings` - List all meetings
- `GET /meeting/<meeting_id>` - Meeting details
- `GET /api/meetings/<meeting_id>/analysis` - Analysis results

### Settings Endpoints
- `GET /api/settings` - Get configuration status
- `POST /api/settings` - Save settings
- `GET /api/settings/test-zoom` - Test Zoom connection
- `GET /api/settings/test-models` - Test ML models

### Health & Monitoring
- `GET /health` - Application health status
- `GET /metrics` - Prometheus metrics

## 🔍 Testing

### Test Zoom Integration
```bash
python test_zoom_setup.py
```

### Test Webhook Endpoint
```bash
curl -X POST http://localhost:5001/webhook/zoom \
  -H "Content-Type: application/json" \
  -d '{"event":"meeting.started","payload":{"object":{"id":"test_123"}}}'
```

### Test ML Pipeline
```bash
python demo_zoom_integration.py
```

### Test MLflow Integration
```bash
python demo_mlflow_integration.py
```

### Test NVIDIA MLOps Stack
```bash
python demo_nvidia_mlops.py
```

### Set Up Monitoring
```bash
./scripts/setup-monitoring.sh
```

## 🚀 CI/CD Pipeline

### **GitHub Actions Workflows**
- **`ci-cd.yml`**: Main CI/CD pipeline with testing, building, and deployment
- **`ml-training.yml`**: Automated ML model training and deployment
- **`security-scan.yml`**: Security vulnerability scanning
- **`deployment.yml`**: Multi-environment deployment automation

### **Pipeline Features**
- ✅ **Code Quality**: Linting, type checking, unit tests
- ✅ **Security Scanning**: Dependency vulnerabilities, code security
- ✅ **ML Testing**: Model validation and performance testing
- ✅ **Docker Builds**: Multi-platform container images
- ✅ **Kubernetes Deployment**: Automated staging and production deployment
- ✅ **Monitoring Setup**: Prometheus and Grafana deployment

**📖 See [CI_CD_SUMMARY.md](CI_CD_SUMMARY.md) for detailed pipeline documentation.**

## 📁 Project Structure

```
vibe-check/
├── app.py                      # Flask web application
├── zoom_integration.py         # Zoom API integration
├── zoom_config.py             # Configuration management
├── mlflow_integration.py      # MLflow model serving & lifecycle
├── triton_integration.py      # Triton GPU serving & NeMo models
├── inference.py               # ML pipeline
├── templates/                 # Web interface templates
│   ├── base.html             # Base template with Bootstrap 5
│   ├── dashboard.html        # Interactive dashboard
│   ├── meetings.html         # Meeting management
│   ├── analytics.html        # Advanced analytics
│   └── settings.html         # Configuration management
├── .github/workflows/         # CI/CD pipelines
│   ├── ci-cd.yml            # Main CI/CD pipeline
│   ├── deployment.yml        # Deployment automation
│   ├── ml-training.yml      # ML training pipeline
│   └── security-scan.yml    # Security scanning
├── models/                    # ML models
│   ├── audio_emotion_model.py
│   ├── sentiment_model.py
│   └── custom/
├── pipeline/                  # ML pipeline components
│   ├── preprocessor.py
│   ├── postprocessor.py
│   └── inference.py
├── utils/                     # Utility functions
├── zoom_recordings/           # Downloaded recordings
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Multi-service Docker setup
├── k8s/                      # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── persistent-volumes.yaml
│   └── ingress.yaml
├── scripts/                   # Deployment scripts
│   ├── deploy-docker.sh
│   ├── deploy-k8s.sh
│   ├── setup-cloud-k8s.sh
│   └── setup-monitoring.sh
├── monitoring/                # Monitoring configuration
│   ├── prometheus.yml
│   ├── rules/
│   │   └── alerts.yml
│   └── grafana/
│       └── dashboards/
│           └── speaking-feedback-dashboard.json
├── deployment-guides/         # Cloud deployment guides
│   ├── aws-eks-setup.md
│   ├── gke-setup.md
│   └── digitalocean-k8s-setup.md
├── .env                      # Environment variables (not in git)
├── .gitignore               # Git ignore rules
├── CI_CD_SUMMARY.md         # CI/CD documentation
├── ENV_SETUP_GUIDE.md       # Environment setup guide
└── README.md                # This file
```

## 🔒 Security

- **No sensitive data stored**: Audio files are deleted after processing
- **Local processing**: All analysis happens on your server
- **Secure webhooks**: HMAC signature verification
- **Environment variables**: Credentials stored securely
- **Security scanning**: Automated vulnerability detection
- **Rate limiting**: Protection against abuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Webhook not receiving events:**
- Check ngrok URL is correct in Zoom app
- Verify webhook endpoint is `/webhook/zoom`
- Check Flask logs for errors

**Audio processing fails:**
- Ensure ffmpeg is installed
- Check file permissions for recordings directory

**ML pipeline errors:**
- Verify model files are present
- Check Python dependencies are installed

**Web interface not loading:**
- Check if Flask app is running on correct port
- Verify all templates are in the templates/ directory
- Check browser console for JavaScript errors

### Getting Help

- Check the [ZOOM_SETUP_GUIDE.md](ZOOM_SETUP_GUIDE.md) for detailed setup instructions
- Review [ZOOM_INTEGRATION_SUMMARY.md](ZOOM_INTEGRATION_SUMMARY.md) for technical details
- See [ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md) for environment configuration
- Review [CI_CD_SUMMARY.md](CI_CD_SUMMARY.md) for pipeline documentation
- Open an issue on GitHub for bugs or feature requests

## 🎯 Roadmap

### **Short Term**
- [ ] Real-time sentiment analysis during meetings
- [ ] Speaker identification and individual feedback
- [ ] Advanced speaking metrics (pace, filler words)
- [ ] Integration with other video platforms

### **Medium Term**
- [ ] Mobile app for feedback viewing
- [ ] Team analytics and reporting
- [ ] Advanced ML models (BERT, GPT)
- [ ] Multi-language support

### **Long Term**
- [ ] AI-powered speaking coach
- [ ] Integration with presentation software
- [ ] Enterprise SSO integration
- [ ] Advanced analytics and insights

## 🏆 Features Overview

| Feature | Status | Description |
|---------|--------|-------------|
| Zoom Integration | ✅ Complete | Real-time webhook processing |
| Web Interface | ✅ Complete | Modern, responsive dashboard |
| ML Pipeline | ✅ Complete | Sentiment and emotion analysis |
| CI/CD Pipeline | ✅ Complete | Automated testing and deployment |
| Docker Support | ✅ Complete | Containerized deployment |
| Kubernetes | ✅ Complete | Production orchestration |
| Monitoring | ✅ Complete | Prometheus & Grafana |
| Security | ✅ Complete | Vulnerability scanning |
| Documentation | ✅ Complete | Comprehensive guides |

---

**Made with ❤️ for better speaking skills**

**Ready for production deployment! 🚀** 