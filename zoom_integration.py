"""
Zoom Integration Module
Handles Zoom webhooks, recording downloads, and meeting processing
"""

import os
import json
import requests
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import tempfile
import subprocess
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZoomIntegration:
    """
    Handles Zoom API integration including webhooks, recording downloads,
    and meeting processing for sentiment analysis
    """
    
    def __init__(self):
        """Initialize Zoom integration with API credentials"""
        self.client_id = os.getenv('ZOOM_CLIENT_ID')
        self.client_secret = os.getenv('ZOOM_CLIENT_SECRET')
        self.webhook_secret = os.getenv('ZOOM_WEBHOOK_SECRET')
        self.account_id = os.getenv('ZOOM_ACCOUNT_ID')
        
        # API endpoints
        self.base_url = "https://api.zoom.us/v2"
        self.auth_url = "https://zoom.us/oauth/token"
        
        # Storage paths
        self.recordings_dir = Path("zoom_recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        # Access token management
        self._access_token = None
        self._token_expires_at = None
        
        logger.info("Zoom integration initialized")
    
    def _get_access_token(self) -> str:
        """Get or refresh Zoom access token"""
        if (self._access_token and self._token_expires_at and 
            datetime.now() < self._token_expires_at):
            return self._access_token
        
        # Get new token
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'account_credentials',
            'account_id': self.account_id
        }
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            self._token_expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'] - 300)  # 5 min buffer
            
            logger.info("Zoom access token refreshed")
            return self._access_token
            
        except Exception as e:
            logger.error(f"Failed to get Zoom access token: {e}")
            raise
    
    def _make_zoom_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request to Zoom API"""
        token = self._get_access_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Zoom API request failed: {e}")
            raise
    
    def verify_webhook_signature(self, payload: str, signature: str, timestamp: str) -> bool:
        """
        Verify webhook signature for security
        
        Args:
            payload: Raw request body
            signature: X-Zoom-Signature header
            timestamp: X-Zoom-Signature-256 header timestamp
            
        Returns:
            bool: True if signature is valid
        """
        if not self.webhook_secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return True
        
        try:
            # Create expected signature
            message = f"v0:{timestamp}:{payload}"
            expected_signature = f"v0={hmac.new(self.webhook_secret.encode(), message.encode(), hashlib.sha256).hexdigest()}"
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def handle_webhook(self, event_data: Dict) -> Dict:
        """
        Handle incoming Zoom webhook events
        
        Args:
            event_data: Parsed webhook payload
            
        Returns:
            Dict: Processing result
        """
        try:
            event_type = event_data.get('event')
            payload = event_data.get('payload', {})
            
            logger.info(f"Processing Zoom webhook: {event_type}")
            
            if event_type == 'recording.completed':
                return self._handle_recording_completed(payload)
            elif event_type == 'meeting.started':
                return self._handle_meeting_started(payload)
            elif event_type == 'meeting.ended':
                return self._handle_meeting_ended(payload)
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return {"status": "ignored", "event_type": event_type}
                
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _handle_recording_completed(self, payload: Dict) -> Dict:
        """Handle recording.completed webhook event"""
        try:
            meeting_id = payload.get('object', {}).get('id')
            recording_files = payload.get('object', {}).get('recording_files', [])
            
            logger.info(f"Recording completed for meeting {meeting_id}")
            
            # Download recordings
            downloaded_files = []
            for recording in recording_files:
                if recording.get('file_type') == 'MP4':  # Video recording
                    file_info = self._download_recording(meeting_id, recording)
                    if file_info:
                        downloaded_files.append(file_info)
            
            # Process the meeting
            if downloaded_files:
                analysis_result = self._process_meeting(meeting_id, downloaded_files)
                return {
                    "status": "success",
                    "meeting_id": meeting_id,
                    "files_downloaded": len(downloaded_files),
                    "analysis_result": analysis_result
                }
            else:
                return {
                    "status": "warning",
                    "meeting_id": meeting_id,
                    "message": "No video recordings found"
                }
                
        except Exception as e:
            logger.error(f"Failed to handle recording completed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _handle_meeting_started(self, payload: Dict) -> Dict:
        """Handle meeting.started webhook event"""
        meeting_id = payload.get('object', {}).get('id')
        logger.info(f"Meeting started: {meeting_id}")
        return {"status": "success", "event": "meeting_started", "meeting_id": meeting_id}
    
    def _handle_meeting_ended(self, payload: Dict) -> Dict:
        """Handle meeting.ended webhook event"""
        meeting_id = payload.get('object', {}).get('id')
        logger.info(f"Meeting ended: {meeting_id}")
        return {"status": "success", "event": "meeting_ended", "meeting_id": meeting_id}
    
    def _download_recording(self, meeting_id: str, recording_info: Dict) -> Optional[Dict]:
        """
        Download a recording file from Zoom
        
        Args:
            meeting_id: Zoom meeting ID
            recording_info: Recording file information
            
        Returns:
            Dict: File information if successful, None otherwise
        """
        try:
            recording_id = recording_info.get('id')
            download_url = recording_info.get('download_url')
            
            if not download_url:
                logger.warning(f"No download URL for recording {recording_id}")
                return None
            
            # Create meeting directory
            meeting_dir = self.recordings_dir / meeting_id
            meeting_dir.mkdir(exist_ok=True)
            
            # Download file
            token = self._get_access_token()
            headers = {'Authorization': f'Bearer {token}'}
            
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Save file
            filename = f"{recording_id}.mp4"
            file_path = meeting_dir / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded recording: {file_path}")
            
            return {
                "recording_id": recording_id,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "download_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to download recording: {e}")
            return None
    
    def _process_meeting(self, meeting_id: str, downloaded_files: List[Dict]) -> Dict:
        """
        Process meeting recordings for sentiment analysis
        
        Args:
            meeting_id: Zoom meeting ID
            downloaded_files: List of downloaded file information
            
        Returns:
            Dict: Analysis results
        """
        try:
            from inference import SentimentAnalysisPipeline
            
            # Initialize pipeline
            pipeline = SentimentAnalysisPipeline()
            
            results = []
            
            for file_info in downloaded_files:
                file_path = file_info['file_path']
                
                # Extract audio from video
                audio_path = self._extract_audio(file_path)
                
                if audio_path:
                    # Analyze audio
                    analysis_result = pipeline.predict_sentiment(
                        audio_path=audio_path,
                        meeting_id=meeting_id,
                        user_id="zoom_user",  # You'll want to map this to actual users
                        team_id="zoom_team"
                    )
                    
                    results.append({
                        "file": file_info['recording_id'],
                        "analysis": analysis_result
                    })
                    
                    # Clean up temporary audio file
                    os.remove(audio_path)
            
            return {
                "meeting_id": meeting_id,
                "files_processed": len(results),
                "results": results,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process meeting {meeting_id}: {e}")
            return {"error": str(e)}
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to video file
            
        Returns:
            str: Path to extracted audio file, None if failed
        """
        try:
            # Check if ffmpeg is available
            if not shutil.which('ffmpeg'):
                logger.warning("ffmpeg not found, skipping audio extraction")
                return None
            
            # Create temporary audio file
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted: {audio_path}")
                return audio_path
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    def get_meeting_info(self, meeting_id: str) -> Optional[Dict]:
        """
        Get meeting information from Zoom API
        
        Args:
            meeting_id: Zoom meeting ID
            
        Returns:
            Dict: Meeting information
        """
        try:
            endpoint = f"/meetings/{meeting_id}"
            return self._make_zoom_request('GET', endpoint)
        except Exception as e:
            logger.error(f"Failed to get meeting info: {e}")
            return None
    
    def list_recordings(self, meeting_id: str) -> Optional[Dict]:
        """
        List recordings for a meeting
        
        Args:
            meeting_id: Zoom meeting ID
            
        Returns:
            Dict: Recording information
        """
        try:
            endpoint = f"/meetings/{meeting_id}/recordings"
            return self._make_zoom_request('GET', endpoint)
        except Exception as e:
            logger.error(f"Failed to list recordings: {e}")
            return None
    
    def get_user_meetings(self, user_id: str, from_date: str = None, to_date: str = None) -> Optional[Dict]:
        """
        Get meetings for a user
        
        Args:
            user_id: Zoom user ID
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict: User meetings
        """
        try:
            params = {}
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
            
            endpoint = f"/users/{user_id}/meetings"
            return self._make_zoom_request('GET', endpoint, params=params)
        except Exception as e:
            logger.error(f"Failed to get user meetings: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test the integration
    zoom = ZoomIntegration()
    
    # Test webhook signature verification
    test_payload = '{"event":"recording.completed","payload":{"object":{"id":"123456789"}}}'
    test_signature = "v0=test_signature"
    test_timestamp = "1234567890"
    
    is_valid = zoom.verify_webhook_signature(test_payload, test_signature, test_timestamp)
    print(f"Signature verification test: {is_valid}")
    
    # Test webhook handling
    test_event = {
        "event": "recording.completed",
        "payload": {
            "object": {
                "id": "123456789",
                "recording_files": [
                    {
                        "id": "recording_123",
                        "file_type": "MP4",
                        "download_url": "https://example.com/recording.mp4"
                    }
                ]
            }
        }
    }
    
    result = zoom.handle_webhook(test_event)
    print(f"Webhook handling test: {result}") 