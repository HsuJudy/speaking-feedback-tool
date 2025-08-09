# 🔧 Environment Variables Setup Guide

## 📋 Overview

The Speaking Feedback Tool uses environment variables stored in a `.env` file for secure configuration. This keeps sensitive credentials out of your code and makes deployment easier.

## 🚀 Quick Setup

### 1. Create `.env` File

Copy the example file and configure your settings:

```bash
cp env_example.txt .env
```

### 2. Configure Zoom Credentials

Edit your `.env` file and add your Zoom App credentials:

```bash
# Zoom API Credentials
ZOOM_CLIENT_ID=your_actual_client_id_here
ZOOM_CLIENT_SECRET=your_actual_client_secret_here
ZOOM_ACCOUNT_ID=your_actual_account_id_here
ZOOM_WEBHOOK_SECRET=your_actual_webhook_secret_here
```

## 🔑 Getting Zoom Credentials

### Step 1: Create Zoom App

1. Go to [Zoom App Marketplace](https://marketplace.zoom.us/)
2. Click "Develop" → "Build App"
3. Choose "General Meeting App"
4. Fill in app details:
   - **App Name**: Speaking Feedback Tool
   - **App Type**: General Meeting App
   - **Publish**: No (for development)

### Step 2: Get Credentials

1. **Client ID & Secret**:
   - Go to "App Credentials" tab
   - Copy the Client ID and Client Secret

2. **Account ID**:
   - Go to "Account" tab
   - Copy the Account ID

3. **Webhook Secret**:
   - Go to "Webhook" tab
   - Add your webhook URL: `https://your-domain.com/webhook/zoom`
   - Copy the Webhook Secret Token

### Step 3: Configure Webhooks

In your Zoom App settings:

1. **Webhook URL**: `https://your-domain.com/webhook/zoom`
2. **Events to Subscribe**:
   - ✅ `meeting.started`
   - ✅ `meeting.ended`
   - ✅ `recording.completed`

## 📝 Complete `.env` Configuration

```bash
# ========================================
# ZOOM API CREDENTIALS
# ========================================
ZOOM_CLIENT_ID=your_client_id_here
ZOOM_CLIENT_SECRET=your_client_secret_here
ZOOM_ACCOUNT_ID=your_account_id_here
ZOOM_WEBHOOK_SECRET=your_webhook_secret_here

# ========================================
# WEBHOOK CONFIGURATION
# ========================================
ZOOM_WEBHOOK_URL=https://your-domain.com/webhook/zoom
ZOOM_APP_NAME=Speaking Feedback Tool

# ========================================
# STORAGE CONFIGURATION
# ========================================
ZOOM_RECORDINGS_DIR=zoom_recordings
ZOOM_MAX_FILE_SIZE=500

# ========================================
# AUDIO PROCESSING
# ========================================
ZOOM_ENABLE_AUDIO_EXTRACTION=true
ZOOM_AUDIO_SAMPLE_RATE=16000
ZOOM_AUDIO_CHANNELS=1

# ========================================
# SECURITY
# ========================================
ZOOM_VERIFY_SIGNATURES=true
ZOOM_ALLOWED_IPS=192.168.1.1,10.0.0.1

# ========================================
# FLASK CONFIGURATION
# ========================================
FLASK_ENV=development
PORT=5000

# ========================================
# OPTIONAL: MLOPS STACK
# ========================================
WANDB_API_KEY=your_wandb_api_key_here
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🔒 Security Best Practices

### ✅ Do's:
- ✅ Store `.env` file in project root
- ✅ Add `.env` to `.gitignore` (already done)
- ✅ Use strong, unique secrets
- ✅ Rotate secrets regularly
- ✅ Use HTTPS in production

### ❌ Don'ts:
- ❌ Never commit `.env` to Git
- ❌ Don't share credentials
- ❌ Don't use default secrets
- ❌ Don't expose `.env` in logs

## 🧪 Testing Configuration

### 1. Check Environment Variables

```bash
python test_zoom_setup.py
```

### 2. Test Zoom Connection

Visit the Settings page in your web interface and click "Test Connection".

### 3. Verify Webhook

Start a Zoom meeting and check if webhooks are received.

## 🚀 Production Deployment

### Environment Variables in Production

For production deployment, set environment variables in your hosting platform:

#### **Heroku:**
```bash
heroku config:set ZOOM_CLIENT_ID=your_client_id
heroku config:set ZOOM_CLIENT_SECRET=your_client_secret
heroku config:set ZOOM_WEBHOOK_SECRET=your_webhook_secret
```

#### **Docker:**
```bash
docker run -e ZOOM_CLIENT_ID=your_client_id \
           -e ZOOM_CLIENT_SECRET=your_client_secret \
           -e ZOOM_WEBHOOK_SECRET=your_webhook_secret \
           your-app
```

#### **Kubernetes:**
```yaml
env:
- name: ZOOM_CLIENT_ID
  valueFrom:
    secretKeyRef:
      name: zoom-secrets
      key: client-id
```

## 🔍 Troubleshooting

### Common Issues:

#### **1. "Zoom: Not Configured"**
- ✅ Check if `.env` file exists
- ✅ Verify `ZOOM_CLIENT_ID` and `ZOOM_CLIENT_SECRET` are set
- ✅ Restart the application after changing `.env`

#### **2. "Webhook: Not Configured"**
- ✅ Verify `ZOOM_WEBHOOK_SECRET` is set
- ✅ Check webhook URL in Zoom App settings

#### **3. "Connection Failed"**
- ✅ Verify credentials are correct
- ✅ Check Zoom App permissions
- ✅ Ensure app is published (for production)

#### **4. "Webhook Not Received"**
- ✅ Verify webhook URL is accessible
- ✅ Check if events are subscribed in Zoom App
- ✅ Ensure webhook secret matches

## 📊 Configuration Status

The web interface shows real-time configuration status:

- 🟢 **Green**: Component configured and ready
- 🔴 **Red**: Component not configured
- 🟡 **Yellow**: Component partially configured

## 🔄 Updating Configuration

### 1. Edit `.env` File
```bash
nano .env
```

### 2. Restart Application
```bash
python app.py
```

### 3. Check Status
Visit the Settings page to verify configuration.

## 📞 Support

If you need help with configuration:

1. **Check the logs**: Look for error messages
2. **Verify credentials**: Double-check Zoom App settings
3. **Test step by step**: Use the test functions in Settings page
4. **Check documentation**: Review Zoom API documentation

---

## 🎯 Next Steps

After configuring your `.env` file:

1. **Test the connection** in the Settings page
2. **Start a Zoom meeting** to test webhooks
3. **Check the dashboard** for real-time data
4. **Deploy to production** when ready

Your Speaking Feedback Tool is now properly configured! 🚀 