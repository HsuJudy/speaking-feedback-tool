# 🎤 Speaking Feedback Tool

A real-time speaking feedback and sentiment analysis tool that integrates with Zoom meetings to provide instant feedback on speaking quality, tone, and sentiment.

## 🚀 Features

- **Real-time Zoom Integration**: Automatically processes Zoom meetings via webhooks
- **Sentiment Analysis**: Analyzes speaking tone and sentiment using ML models
- **Audio Processing**: Extracts and processes audio from Zoom recordings
- **Dashboard**: Web interface to view analysis results and meeting history
- **Multi-Model Support**: Supports various ML models (HuggingFace, NeMo, custom models)
- **Secure**: Local processing with automatic cleanup of sensitive data

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
```

## 📋 Prerequisites

- Python 3.8+
- Zoom Developer Account
- ngrok (for local development)
- ffmpeg (for audio processing)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Speaking-Feedback-Tool/vibe-check
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp env_example.txt .env
# Edit .env with your Zoom credentials
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
```

## 🚀 Usage

### Local Development

1. **Start the Flask App:**
```bash
python app.py
```

2. **Start ngrok (for webhook testing):**
```bash
ngrok http 5001
```

3. **Update Zoom webhook URL** with your ngrok URL:
```
https://your-ngrok-url.ngrok-free.app/webhook/zoom
```

### Production Deployment

#### Option 1: Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
heroku config:set ZOOM_CLIENT_ID=your_client_id
# ... set other environment variables
```

#### Option 2: Railway
```bash
# Connect your GitHub repo to Railway
# Set environment variables in Railway dashboard
```

#### Option 3: DigitalOcean App Platform
```bash
# Connect your GitHub repo to DigitalOcean
# Set environment variables in App Platform dashboard
```

## 📊 API Endpoints

### Webhook Endpoints
- `POST /webhook/zoom` - Zoom webhook receiver
- `GET /oauth/callback` - OAuth callback handler

### Dashboard Endpoints
- `GET /` - Main dashboard
- `GET /meetings` - List all meetings
- `GET /meeting/<meeting_id>` - Meeting details
- `GET /api/meetings/<meeting_id>/analysis` - Analysis results

### Health Check
- `GET /health` - Application health status

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

## 📁 Project Structure

```
vibe-check/
├── app.py                      # Flask web application
├── zoom_integration.py         # Zoom API integration
├── zoom_config.py             # Configuration management
├── inference.py               # ML pipeline
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
├── .env                      # Environment variables (not in git)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔒 Security

- **No sensitive data stored**: Audio files are deleted after processing
- **Local processing**: All analysis happens on your server
- **Secure webhooks**: HMAC signature verification
- **Environment variables**: Credentials stored securely

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

### Getting Help

- Check the [ZOOM_SETUP_GUIDE.md](ZOOM_SETUP_GUIDE.md) for detailed setup instructions
- Review [ZOOM_INTEGRATION_SUMMARY.md](ZOOM_INTEGRATION_SUMMARY.md) for technical details
- Open an issue on GitHub for bugs or feature requests

## 🎯 Roadmap

- [ ] Real-time sentiment analysis during meetings
- [ ] Speaker identification and individual feedback
- [ ] Advanced speaking metrics (pace, filler words)
- [ ] Integration with other video platforms
- [ ] Mobile app for feedback viewing
- [ ] Team analytics and reporting

---

**Made with ❤️ for better speaking skills** 