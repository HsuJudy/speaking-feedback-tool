"""
Demo: Zoom Integration
Test and demonstrate Zoom integration functionality
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_zoom_integration():
    """Demonstrate Zoom integration functionality"""
    
    print("🎤 DEMO: Zoom Integration")
    print("=" * 50)
    
    # Import Zoom components
    try:
        from zoom_integration import ZoomIntegration
        from zoom_config import ZoomConfig
        print("✅ Zoom modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Zoom modules: {e}")
        return False
    
    # Test configuration
    print("\n🔧 Testing Configuration")
    print("-" * 30)
    
    config = ZoomConfig()
    validation = config.validate_config()
    
    print(f"Configuration valid: {validation['valid']}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Client ID set: {validation['config']['client_id_set']}")
    print(f"  Client Secret set: {validation['config']['client_secret_set']}")
    print(f"  Account ID set: {validation['config']['account_id_set']}")
    print(f"  Webhook Secret set: {validation['config']['webhook_secret_set']}")
    print(f"  Recordings directory: {validation['config']['recordings_dir']}")
    print(f"  Webhook URL: {validation['config']['webhook_url']}")
    print(f"  Verify signatures: {validation['config']['verify_signatures']}")
    
    # Test Zoom integration initialization
    print("\n🔗 Testing Zoom Integration")
    print("-" * 30)
    
    try:
        zoom = ZoomIntegration()
        print("✅ Zoom integration initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Zoom integration: {e}")
        return False
    
    # Test webhook signature verification
    print("\n🔐 Testing Webhook Security")
    print("-" * 30)
    
    test_payload = '{"event":"recording.completed","payload":{"object":{"id":"123456789"}}}'
    test_signature = "v0=test_signature"
    test_timestamp = "1234567890"
    
    is_valid = zoom.verify_webhook_signature(test_payload, test_signature, test_timestamp)
    print(f"Signature verification test: {is_valid}")
    
    # Test webhook handling
    print("\n📡 Testing Webhook Handling")
    print("-" * 30)
    
    test_events = [
        {
            "event": "meeting.started",
            "payload": {
                "object": {
                    "id": "123456789",
                    "topic": "Test Meeting",
                    "start_time": "2024-01-01T10:00:00Z"
                }
            }
        },
        {
            "event": "meeting.ended",
            "payload": {
                "object": {
                    "id": "123456789",
                    "topic": "Test Meeting",
                    "end_time": "2024-01-01T11:00:00Z"
                }
            }
        },
        {
            "event": "recording.completed",
            "payload": {
                "object": {
                    "id": "123456789",
                    "recording_files": [
                        {
                            "id": "recording_123",
                            "file_type": "MP4",
                            "download_url": "https://example.com/recording.mp4",
                            "file_size": 1024000
                        }
                    ]
                }
            }
        }
    ]
    
    for i, event in enumerate(test_events, 1):
        print(f"\nTest {i}: {event['event']}")
        try:
            result = zoom.handle_webhook(event)
            print(f"  Status: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            elif 'message' in result:
                print(f"  Message: {result['message']}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Test API methods (if credentials are available)
    print("\n🌐 Testing API Methods")
    print("-" * 30)
    
    if validation['config']['client_id_set']:
        print("Testing API methods with credentials...")
        
        # Test getting meeting info
        test_meeting_id = "123456789"
        try:
            meeting_info = zoom.get_meeting_info(test_meeting_id)
            if meeting_info:
                print(f"✅ Meeting info retrieved for {test_meeting_id}")
            else:
                print(f"⚠️  No meeting info found for {test_meeting_id}")
        except Exception as e:
            print(f"❌ Failed to get meeting info: {e}")
        
        # Test listing recordings
        try:
            recordings = zoom.list_recordings(test_meeting_id)
            if recordings:
                print(f"✅ Recordings listed for {test_meeting_id}")
            else:
                print(f"⚠️  No recordings found for {test_meeting_id}")
        except Exception as e:
            print(f"❌ Failed to list recordings: {e}")
    
    else:
        print("⚠️  Skipping API tests - no credentials configured")
    
    # Test audio extraction
    print("\n🎵 Testing Audio Extraction")
    print("-" * 30)
    
    # Create a dummy video file for testing
    test_video_path = "test_video.mp4"
    
    # Check if ffmpeg is available
    import shutil
    if shutil.which('ffmpeg'):
        print("✅ ffmpeg found - audio extraction available")
        
        # Create a simple test video (if possible)
        try:
            import subprocess
            # Create a simple test video using ffmpeg
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=5:size=320x240:rate=1',
                '-f', 'lavfi', '-i', 'sine=frequency=440:duration=5',
                '-c:v', 'libx264', '-c:a', 'aac', '-y', test_video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and Path(test_video_path).exists():
                print("✅ Test video created")
                
                # Test audio extraction
                audio_path = zoom._extract_audio(test_video_path)
                if audio_path:
                    print(f"✅ Audio extracted: {audio_path}")
                    
                    # Clean up test files
                    if Path(test_video_path).exists():
                        Path(test_video_path).unlink()
                    if Path(audio_path).exists():
                        Path(audio_path).unlink()
                else:
                    print("❌ Audio extraction failed")
            else:
                print("⚠️  Could not create test video")
                
        except Exception as e:
            print(f"❌ Audio extraction test failed: {e}")
    else:
        print("⚠️  ffmpeg not found - audio extraction not available")
    
    # Test Flask app (if available)
    print("\n🌐 Testing Flask App")
    print("-" * 30)
    
    try:
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test health endpoint
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint working")
                health_data = response.get_json()
                print(f"  Status: {health_data.get('status')}")
                print(f"  Zoom Integration: {health_data.get('zoom_integration')}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test meetings endpoint
        with app.test_client() as client:
            response = client.get('/meetings')
            if response.status_code == 200:
                print("✅ Meetings endpoint working")
                meetings_data = response.get_json()
                print(f"  Meetings found: {len(meetings_data.get('meetings', []))}")
            else:
                print(f"❌ Meetings endpoint failed: {response.status_code}")
        
    except ImportError as e:
        print(f"⚠️  Flask app not available: {e}")
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
    
    # Show setup instructions
    print("\n📋 Setup Instructions")
    print("-" * 30)
    
    config.print_setup_instructions()
    
    print("\n🎉 Zoom Integration Demo Complete!")
    print("=" * 50)
    
    return True

def demo_webhook_simulation():
    """Simulate webhook processing"""
    
    print("\n🔗 DEMO: Webhook Simulation")
    print("=" * 40)
    
    try:
        from zoom_integration import ZoomIntegration
        from app import app
        
        zoom = ZoomIntegration()
        
        # Simulate webhook payload
        webhook_payload = {
            "event": "recording.completed",
            "payload": {
                "object": {
                    "id": "demo_meeting_123",
                    "topic": "Demo Meeting",
                    "start_time": "2024-01-01T10:00:00Z",
                    "end_time": "2024-01-01T11:00:00Z",
                    "duration": 60,
                    "recording_files": [
                        {
                            "id": "demo_recording_123",
                            "file_type": "MP4",
                            "download_url": "https://example.com/demo_recording.mp4",
                            "file_size": 2048000
                        }
                    ]
                }
            }
        }
        
        print("Simulating webhook processing...")
        
        # Process webhook
        result = zoom.handle_webhook(webhook_payload)
        
        print(f"Webhook processing result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Meeting ID: {result.get('meeting_id')}")
        
        if 'analysis_result' in result:
            analysis = result['analysis_result']
            print(f"  Files processed: {analysis.get('files_processed')}")
            print(f"  Processed at: {analysis.get('processed_at')}")
        
        # Test Flask webhook endpoint
        with app.test_client() as client:
            # Create test request
            headers = {
                'Content-Type': 'application/json',
                'X-Zoom-Signature': 'v0=test_signature',
                'X-Zoom-Signature-256': 'v0=test_signature'
            }
            
            response = client.post('/webhook/zoom', 
                                json=webhook_payload,
                                headers=headers)
            
            print(f"\nFlask webhook endpoint response:")
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {response.get_json()}")
        
        print("✅ Webhook simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Webhook simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🎤 Zoom Integration Demo")
    print("=" * 50)
    
    # Run main demo
    success = demo_zoom_integration()
    
    if success:
        # Run webhook simulation
        demo_webhook_simulation()
    
    print("\n🎯 Next Steps:")
    print("1. Set up Zoom App in Marketplace")
    print("2. Configure environment variables")
    print("3. Deploy webhook endpoint")
    print("4. Test with real Zoom meetings")
    print("5. Monitor analysis results") 