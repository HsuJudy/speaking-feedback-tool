"""
Zoom Configuration
Settings and environment variables for Zoom integration
"""

import os
from typing import Dict, Any
from pathlib import Path

class ZoomConfig:
    """Configuration class for Zoom integration"""
    
    def __init__(self):
        """Initialize Zoom configuration"""
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables"""
        # Required Zoom API credentials
        self.client_id = os.getenv('ZOOM_CLIENT_ID')
        self.client_secret = os.getenv('ZOOM_CLIENT_SECRET')
        self.account_id = os.getenv('ZOOM_ACCOUNT_ID')
        self.webhook_secret = os.getenv('ZOOM_WEBHOOK_SECRET')
        
        # Optional settings
        self.webhook_url = os.getenv('ZOOM_WEBHOOK_URL', 'https://your-domain.com/webhook/zoom')
        self.app_name = os.getenv('ZOOM_APP_NAME', 'Speaking Feedback Tool')
        
        # Storage settings
        self.recordings_dir = Path(os.getenv('ZOOM_RECORDINGS_DIR', 'zoom_recordings'))
        self.max_file_size = int(os.getenv('ZOOM_MAX_FILE_SIZE', '500'))  # MB
        
        # Processing settings
        self.enable_audio_extraction = os.getenv('ZOOM_ENABLE_AUDIO_EXTRACTION', 'true').lower() == 'true'
        self.audio_sample_rate = int(os.getenv('ZOOM_AUDIO_SAMPLE_RATE', '16000'))
        self.audio_channels = int(os.getenv('ZOOM_AUDIO_CHANNELS', '1'))
        
        # Webhook settings
        self.webhook_events = [
            'meeting.started',
            'meeting.ended', 
            'recording.completed',
            'meeting.participant_joined',
            'meeting.participant_left'
        ]
        
        # Security settings
        self.verify_signatures = os.getenv('ZOOM_VERIFY_SIGNATURES', 'true').lower() == 'true'
        self.allowed_ips = os.getenv('ZOOM_ALLOWED_IPS', '').split(',') if os.getenv('ZOOM_ALLOWED_IPS') else []
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration and return status
        
        Returns:
            Dict: Configuration validation results
        """
        errors = []
        warnings = []
        
        # Check required credentials
        if not self.client_id:
            errors.append("ZOOM_CLIENT_ID is required")
        if not self.client_secret:
            errors.append("ZOOM_CLIENT_SECRET is required")
        if not self.account_id:
            errors.append("ZOOM_ACCOUNT_ID is required")
        
        # Check optional settings
        if not self.webhook_secret:
            warnings.append("ZOOM_WEBHOOK_SECRET not set - signature verification disabled")
        
        if not self.webhook_url.startswith('https://'):
            warnings.append("Webhook URL should use HTTPS for production")
        
        # Check storage directory
        try:
            self.recordings_dir.mkdir(exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create recordings directory: {e}")
        
        # Check ffmpeg availability
        import shutil
        if not shutil.which('ffmpeg'):
            warnings.append("ffmpeg not found - audio extraction will be disabled")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config": {
                "client_id_set": bool(self.client_id),
                "client_secret_set": bool(self.client_secret),
                "account_id_set": bool(self.account_id),
                "webhook_secret_set": bool(self.webhook_secret),
                "recordings_dir": str(self.recordings_dir),
                "webhook_url": self.webhook_url,
                "verify_signatures": self.verify_signatures
            }
        }
    
    def get_zoom_app_config(self) -> Dict[str, Any]:
        """
        Get configuration for Zoom App Marketplace
        
        Returns:
            Dict: Zoom App configuration
        """
        return {
            "name": self.app_name,
            "type": "Meeting",
            "description": "Real-time speaking feedback and sentiment analysis for Zoom meetings",
            "webhook_url": self.webhook_url,
            "webhook_events": self.webhook_events,
            "permissions": [
                "meeting:read",
                "meeting:write",
                "recording:read",
                "user:read"
            ],
            "scopes": [
                "meeting:read",
                "meeting:write",
                "recording:read",
                "user:read"
            ]
        }
    
    def get_environment_template(self) -> str:
        """
        Get environment variables template
        
        Returns:
            str: Environment variables template
        """
        return """
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
"""
    
    def print_setup_instructions(self):
        """Print setup instructions for Zoom integration"""
        print("🔧 Zoom Integration Setup Instructions")
        print("=" * 50)
        print()
        
        print("1. Create Zoom App:")
        print("   - Go to https://marketplace.zoom.us/")
        print("   - Click 'Develop' → 'Build App'")
        print("   - Choose 'Meeting' app type")
        print("   - Fill in app details:")
        print(f"     * Name: {self.app_name}")
        print("     * Description: Real-time speaking feedback and sentiment analysis")
        print()
        
        print("2. Configure App Settings:")
        print("   - In 'Meeting' tab, add webhook URL:")
        print(f"     * URL: {self.webhook_url}")
        print("   - Add webhook events:")
        for event in self.webhook_events:
            print(f"     * {event}")
        print()
        
        print("3. Get Credentials:")
        print("   - Copy Client ID and Client Secret from 'App Credentials'")
        print("   - Copy Account ID from 'Account' tab")
        print("   - Copy Webhook Secret from 'Webhook' tab")
        print()
        
        print("4. Set Environment Variables:")
        print("   Create a .env file with:")
        print(self.get_environment_template())
        print()
        
        print("5. Deploy Webhook Endpoint:")
        print("   - Deploy to Heroku, Railway, or your preferred platform")
        print("   - Update ZOOM_WEBHOOK_URL with your production URL")
        print()
        
        print("6. Test Integration:")
        print("   - Start a Zoom meeting with recording enabled")
        print("   - End the meeting")
        print("   - Check webhook logs for processing")
        print()
        
        validation = self.validate_config()
        if validation["valid"]:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration errors found:")
            for error in validation["errors"]:
                print(f"   - {error}")
        
        if validation["warnings"]:
            print("⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")


# Example usage
if __name__ == "__main__":
    config = ZoomConfig()
    config.print_setup_instructions() 