"""
Test Zoom Setup
Simple script to test Zoom integration setup
"""

import os
from dotenv import load_dotenv

def test_zoom_setup():
    """Test Zoom setup and provide guidance"""
    
    print("🔧 Testing Zoom Setup")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    required_vars = [
        'ZOOM_CLIENT_ID',
        'ZOOM_CLIENT_SECRET', 
        'ZOOM_ACCOUNT_ID',
        'ZOOM_WEBHOOK_SECRET'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {'*' * len(value)}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        
        print(f"\n📝 To fix this, create a .env file with:")
        print("""
# Zoom API Credentials
ZOOM_CLIENT_ID=your_client_id_here
ZOOM_CLIENT_SECRET=your_client_secret_here
ZOOM_ACCOUNT_ID=your_account_id_here
ZOOM_WEBHOOK_SECRET=your_webhook_secret_here

# Optional settings
ZOOM_WEBHOOK_URL=https://your-domain.com/webhook/zoom
ZOOM_APP_NAME=Speaking Feedback Tool
        """)
        
        print("🔗 Get these credentials from:")
        print("   https://marketplace.zoom.us/develop/create")
        
        return False
    else:
        print("\n✅ All required environment variables are set!")
        
        # Test Zoom integration
        try:
            from zoom_integration import ZoomIntegration
            zoom = ZoomIntegration()
            print("✅ Zoom integration initialized successfully")
            
            # Test configuration
            from zoom_config import ZoomConfig
            config = ZoomConfig()
            validation = config.validate_config()
            
            if validation['valid']:
                print("✅ Configuration is valid!")
                return True
            else:
                print("❌ Configuration has errors:")
                for error in validation['errors']:
                    print(f"   - {error}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to initialize Zoom integration: {e}")
            return False

if __name__ == "__main__":
    success = test_zoom_setup()
    
    if success:
        print("\n🎉 Zoom setup is ready!")
        print("Next steps:")
        print("1. Deploy your webhook endpoint")
        print("2. Update ZOOM_WEBHOOK_URL in your .env file")
        print("3. Test with a real Zoom meeting")
    else:
        print("\n⚠️  Please fix the issues above before proceeding") 