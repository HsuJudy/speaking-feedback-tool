# 🎤 Zoom Integration Setup Guide

Complete guide to set up Zoom integration for the Speaking Feedback Tool.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Zoom App Setup](#zoom-app-setup)
3. [Environment Configuration](#environment-configuration)
4. [Local Development](#local-development)
5. [Production Deployment](#production-deployment)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Software
- Python 3.8+
- ffmpeg (for audio extraction)
- Git

### Required Accounts
- Zoom Developer Account
- Heroku/Railway account (for deployment)

### Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg

# Install ffmpeg (Windows)
# Download from https://ffmpeg.org/download.html
```

---

## 🏗️ Zoom App Setup

### Step 1: Create Zoom App

1. Go to [Zoom App Marketplace](https://marketplace.zoom.us/)
2. Click **"Develop"** → **"Build App"**
3. Choose **"Meeting"** app type
4. Fill in app details:
   - **Name**: Speaking Feedback Tool
   - **Description**: Real-time speaking feedback and sentiment analysis
   - **Category**: Productivity

### Step 2: Configure App Settings

#### Meeting Tab
1. Click **"Meeting"** tab
2. Add **Webhook URL**:
   ```
   https://your-domain.com/webhook/zoom
   ```
3. Add **Webhook Events**:
   - `meeting.started`
   - `meeting.ended`
   - `recording.completed`
   - `meeting.participant_joined`
   - `meeting.participant_left`

#### Permissions Tab
1. Click **"Permissions"** tab
2. Add **Meeting** permissions:
   - `meeting:read`
   - `meeting:write`
   - `recording:read`
   - `user:read`

### Step 3: Get Credentials

#### App Credentials
1. Click **"App Credentials"** tab
2. Copy **Client ID** and **Client Secret**

#### Account Tab
1. Click **"Account"** tab
2. Copy **Account ID**

#### Webhook Tab
1. Click **"Webhook"** tab
2. Copy **Webhook Secret**

---

## ⚙️ Environment Configuration

### Step 1: Create Environment File

Create `.env` file in the project root:

```bash
# Zoom API Credentials
ZOOM_CLIENT_ID=your_client_id_here
ZOOM_CLIENT_SECRET=your_client_secret_here
ZOOM_ACCOUNT_ID=your_account_id_here
ZOOM_WEBHOOK_SECRET=your_webhook_secret_here

# Webhook Configuration
ZOOM_WEBHOOK_URL=https://your-domain.com/webhook/zoom
ZOOM_APP_NAME=Speaking Feedback Tool

# Storage Configuration
ZOOM_RECORDINGS_DIR=zoom_recordings
ZOOM_MAX_FILE_SIZE=500

# Audio Processing
ZOOM_ENABLE_AUDIO_EXTRACTION=true
ZOOM_AUDIO_SAMPLE_RATE=16000
ZOOM_AUDIO_CHANNELS=1

# Security
ZOOM_VERIFY_SIGNATURES=true
ZOOM_ALLOWED_IPS=192.168.1.1,10.0.0.1

# Flask Configuration
FLASK_ENV=development
PORT=5000
```

### Step 2: Load Environment Variables

```python
# In your Python code
from dotenv import load_dotenv
load_dotenv()
```

---

## 🚀 Local Development

### Step 1: Test Configuration

```bash
# Test Zoom configuration
python zoom_config.py
```

### Step 2: Run Demo

```bash
# Run Zoom integration demo
python demo_zoom_integration.py
```

### Step 3: Start Flask App

```bash
# Start local development server
python app.py
```

The app will be available at `http://localhost:5000`

### Step 4: Test Webhook (Optional)

For local testing, you can use ngrok to expose your local server:

```bash
# Install ngrok
brew install ngrok  # macOS
# or download from https://ngrok.com/

# Expose local server
ngrok http 5000
```

Update your Zoom webhook URL to the ngrok URL:
```
https://your-ngrok-url.ngrok.io/webhook/zoom
```

---

## 🌐 Production Deployment

### Option 1: Heroku

#### Step 1: Create Heroku App

```bash
# Install Heroku CLI
# Download from https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create app
heroku create your-speaking-feedback-app

# Add buildpack for ffmpeg
heroku buildpacks:add https://github.com/heroku/heroku-buildpack-ffmpeg-latest.git
heroku buildpacks:add heroku/python
```

#### Step 2: Configure Environment Variables

```bash
# Set environment variables
heroku config:set ZOOM_CLIENT_ID=your_client_id
heroku config:set ZOOM_CLIENT_SECRET=your_client_secret
heroku config:set ZOOM_ACCOUNT_ID=your_account_id
heroku config:set ZOOM_WEBHOOK_SECRET=your_webhook_secret
heroku config:set ZOOM_WEBHOOK_URL=https://your-app-name.herokuapp.com/webhook/zoom
```

#### Step 3: Deploy

```bash
# Deploy to Heroku
git add .
git commit -m "Add Zoom integration"
git push heroku main
```

### Option 2: Railway

#### Step 1: Create Railway Project

1. Go to [Railway](https://railway.app/)
2. Create new project
3. Connect your GitHub repository

#### Step 2: Configure Environment Variables

In Railway dashboard:
1. Go to your project
2. Click **"Variables"** tab
3. Add all environment variables from `.env` file

#### Step 3: Deploy

Railway will automatically deploy when you push to GitHub.

### Option 3: DigitalOcean App Platform

#### Step 1: Create App

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Create new app
3. Connect your GitHub repository

#### Step 2: Configure

1. Set build command: `pip install -r requirements.txt`
2. Set run command: `python app.py`
3. Add environment variables

#### Step 3: Deploy

DigitalOcean will automatically deploy your app.

---

## 🧪 Testing

### Step 1: Test Configuration

```bash
# Test Zoom configuration
python zoom_config.py
```

Expected output:
```
✅ Configuration is valid!
```

### Step 2: Test Integration

```bash
# Run integration demo
python demo_zoom_integration.py
```

### Step 3: Test Webhook

1. Start a Zoom meeting with recording enabled
2. End the meeting
3. Check webhook logs in your deployment platform
4. Verify analysis results in the dashboard

### Step 4: Test API Endpoints

```bash
# Health check
curl https://your-domain.com/health

# List meetings
curl https://your-domain.com/meetings

# Get meeting details
curl https://your-domain.com/meeting/meeting_id
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "Invalid webhook signature"

**Cause**: Webhook secret mismatch
**Solution**: 
- Verify webhook secret in Zoom App settings
- Check environment variable `ZOOM_WEBHOOK_SECRET`

#### 2. "ffmpeg not found"

**Cause**: ffmpeg not installed
**Solution**:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### 3. "Failed to get Zoom access token"

**Cause**: Invalid credentials
**Solution**:
- Verify `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET`, `ZOOM_ACCOUNT_ID`
- Check Zoom App status in marketplace

#### 4. "Recording download failed"

**Cause**: Insufficient permissions
**Solution**:
- Verify recording permissions in Zoom App
- Check meeting recording settings

#### 5. "Audio extraction failed"

**Cause**: Video file format issues
**Solution**:
- Ensure ffmpeg is installed
- Check video file format compatibility
- Verify file permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Logs

#### Heroku
```bash
heroku logs --tail
```

#### Railway
- Check logs in Railway dashboard

#### Local
```bash
python app.py
# Check console output
```

---

## 📊 Monitoring

### Health Check

Monitor your application health:

```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T10:00:00Z",
  "zoom_integration": "active"
}
```

### Dashboard

Access the web dashboard at:
```
https://your-domain.com/
```

### Database

Check analysis results:

```bash
# Connect to database (if using SQLite)
sqlite3 zoom_analysis.db

# View tables
.tables

# Check meetings
SELECT * FROM meetings;

# Check analysis results
SELECT * FROM analysis_results;
```

---

## 🔐 Security Best Practices

### 1. Environment Variables
- Never commit `.env` files to version control
- Use secure secret management in production
- Rotate secrets regularly

### 2. Webhook Security
- Always verify webhook signatures
- Use HTTPS in production
- Implement rate limiting

### 3. Data Privacy
- Encrypt sensitive data
- Implement data retention policies
- Comply with GDPR/privacy regulations

### 4. Access Control
- Implement user authentication
- Use role-based access control
- Audit access logs

---

## 📈 Next Steps

### 1. User Management
- Implement user registration/login
- Add user-specific meeting access
- Create user dashboards

### 2. Advanced Analytics
- Add real-time processing
- Implement custom model training
- Add more emotion detection

### 3. Integration Features
- Add Slack notifications
- Implement email reports
- Create mobile app

### 4. Business Features
- Add billing integration
- Implement team management
- Create API documentation

---

## 🆘 Support

### Documentation
- [Zoom API Documentation](https://marketplace.zoom.us/docs/api-reference/zoom-api)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)

### Community
- [Zoom Developer Community](https://devforum.zoom.us/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/zoom-api)

### Issues
- Check the troubleshooting section above
- Review application logs
- Test with minimal configuration

---

## ✅ Checklist

- [ ] Zoom App created in marketplace
- [ ] Webhook URL configured
- [ ] Permissions set correctly
- [ ] Credentials copied to environment
- [ ] Local development working
- [ ] Production deployment complete
- [ ] Webhook testing successful
- [ ] Analysis pipeline working
- [ ] Dashboard accessible
- [ ] Monitoring configured

---

**🎉 Congratulations!** Your Zoom integration is now set up and ready to analyze meeting recordings for sentiment and stress detection. 