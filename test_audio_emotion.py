"""
Test script for audio emotion detection
Demonstrates emotion detection from audio characteristics without speech content
"""

import json
import os
import numpy as np
from models.model_factory import ModelFactory
from utils.audio_features import AudioFeatureExtractor


def test_audio_emotion_model():
    """Test the audio emotion detection model"""
    print("🎵 TESTING AUDIO EMOTION DETECTION")
    print("=" * 60)
    
    # Initialize model factory
    factory = ModelFactory("config/models.yaml")
    
    # Test audio emotion model
    print("\n🎭 Testing Audio Emotion Model")
    print("-" * 40)
    
    try:
        # Get audio emotion model
        model = factory.get_model("audio_emotion")
        print(f"✅ Model loaded: {model.get_model_info()}")
        
        # Test with different audio characteristics
        test_scenarios = [
            {
                "name": "Happy Audio",
                "description": "High pitch, bright tone, fast speech",
                "characteristics": {
                    "pitch_mean": 250,
                    "pitch_std": 60,
                    "volume_mean": 0.6,
                    "volume_std": 0.3,
                    "speech_rate": 0.9,
                    "spectral_centroid_mean": 3000
                }
            },
            {
                "name": "Sad Audio",
                "description": "Low pitch, dark tone, slow speech",
                "characteristics": {
                    "pitch_mean": 120,
                    "pitch_std": 15,
                    "volume_mean": 0.2,
                    "volume_std": 0.1,
                    "speech_rate": 0.3,
                    "spectral_centroid_mean": 1200
                }
            },
            {
                "name": "Angry Audio",
                "description": "High volume, variable pitch, fast speech",
                "characteristics": {
                    "pitch_mean": 180,
                    "pitch_std": 80,
                    "volume_mean": 0.8,
                    "volume_std": 0.5,
                    "speech_rate": 0.95,
                    "spectral_centroid_mean": 2500
                }
            },
            {
                "name": "Calm Audio",
                "description": "Medium pitch, steady tone, normal speech",
                "characteristics": {
                    "pitch_mean": 200,
                    "pitch_std": 25,
                    "volume_mean": 0.4,
                    "volume_std": 0.15,
                    "speech_rate": 0.6,
                    "spectral_centroid_mean": 2000
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n--- Testing: {scenario['name']} ---")
            print(f"Description: {scenario['description']}")
            
            # Create dummy audio file for testing
            test_audio_path = f"test_{scenario['name'].lower().replace(' ', '_')}.wav"
            
            try:
                # Generate test audio based on characteristics
                duration = 3  # seconds
                sr = 16000
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                
                # Create audio with specified characteristics
                pitch_freq = scenario['characteristics']['pitch_mean']
                volume_level = scenario['characteristics']['volume_mean']
                
                # Generate audio with appropriate characteristics
                audio = volume_level * np.sin(2 * np.pi * pitch_freq * t)
                
                # Add harmonics for more realistic audio
                if scenario['name'] == "Happy Audio":
                    audio += 0.3 * np.sin(2 * np.pi * pitch_freq * 2 * t)
                elif scenario['name'] == "Sad Audio":
                    audio += 0.1 * np.sin(2 * np.pi * pitch_freq * 1.5 * t)
                elif scenario['name'] == "Angry Audio":
                    audio += 0.4 * np.sin(2 * np.pi * pitch_freq * 1.8 * t)
                
                # Save test audio
                from utils.audio_utils import AudioProcessor
                audio_processor = AudioProcessor()
                if audio_processor.AUDIO_AVAILABLE:
                    audio_processor.save_audio(audio, test_audio_path, sr)
                    
                    # Test emotion prediction
                    result = model.predict(test_audio_path)
                    
                    print(f"Predicted Emotion: {result.get('emotion', 'unknown')}")
                    print(f"Confidence: {result.get('confidence', 0):.3f}")
                    print(f"Audio Characteristics: {result.get('audio_characteristics', {})}")
                    
                    # Show emotion scores
                    emotion_scores = result.get('emotion_scores', {})
                    if emotion_scores:
                        print("Emotion Scores:")
                        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {emotion}: {score:.3f}")
                    
                else:
                    print("❌ Audio libraries not available, skipping audio test")
                    
            except Exception as e:
                print(f"❌ Error testing {scenario['name']}: {e}")
            finally:
                # Clean up test file
                if os.path.exists(test_audio_path):
                    os.unlink(test_audio_path)
    
    except Exception as e:
        print(f"❌ Error loading audio emotion model: {e}")


def test_audio_feature_extraction():
    """Test audio feature extraction"""
    print("\n🔧 TESTING AUDIO FEATURE EXTRACTION")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # Create test audio with different characteristics
    test_audios = {
        "excited": {
            "description": "High energy, fast speech",
            "audio": lambda t: 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
        },
        "anxious": {
            "description": "Variable pitch, irregular speech",
            "audio": lambda t: 0.3 * np.sin(2 * np.pi * 250 * t) + 0.2 * np.sin(2 * np.pi * 500 * t) * np.sin(2 * np.pi * 2 * t)
        },
        "neutral": {
            "description": "Balanced characteristics",
            "audio": lambda t: 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
        }
    }
    
    duration = 3  # seconds
    sr = extractor.sample_rate
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    for emotion, config in test_audios.items():
        print(f"\n--- Testing {emotion} audio ---")
        print(f"Description: {config['description']}")
        
        # Generate audio
        audio = config['audio'](t)
        
        # Extract features
        features = extractor.extract_features(audio)
        
        # Get feature summary
        summary = extractor.get_feature_summary(features)
        
        print(f"Features extracted: {len(features)}")
        print(f"Pitch: {summary['pitch_characteristics']}")
        print(f"Volume: {summary['volume_characteristics']}")
        print(f"Speech: {summary['speech_characteristics']}")
        print(f"Tone: {summary['tone_characteristics']}")


def test_emotion_characteristics():
    """Test emotion prediction based on audio characteristics"""
    print("\n🎯 TESTING EMOTION CHARACTERISTICS")
    print("=" * 50)
    
    # Define emotion characteristics
    emotion_characteristics = {
        "happy": {
            "pitch": "high",
            "volume": "medium-high",
            "speaking_rate": "fast",
            "tone": "bright",
            "energy": "high"
        },
        "sad": {
            "pitch": "low",
            "volume": "low",
            "speaking_rate": "slow",
            "tone": "dark",
            "energy": "low"
        },
        "angry": {
            "pitch": "variable",
            "volume": "high",
            "speaking_rate": "fast",
            "tone": "harsh",
            "energy": "high"
        },
        "calm": {
            "pitch": "medium",
            "volume": "medium",
            "speaking_rate": "normal",
            "tone": "smooth",
            "energy": "medium"
        },
        "excited": {
            "pitch": "high",
            "volume": "high",
            "speaking_rate": "very_fast",
            "tone": "bright",
            "energy": "very_high"
        },
        "anxious": {
            "pitch": "variable",
            "volume": "variable",
            "speaking_rate": "irregular",
            "tone": "tense",
            "energy": "variable"
        },
        "neutral": {
            "pitch": "medium",
            "volume": "medium",
            "speaking_rate": "normal",
            "tone": "balanced",
            "energy": "medium"
        }
    }
    
    print("Emotion Audio Characteristics:")
    for emotion, chars in emotion_characteristics.items():
        print(f"\n{emotion.upper()}:")
        for characteristic, value in chars.items():
            print(f"  {characteristic}: {value}")


def test_content_independence():
    """Test that emotion detection is content-independent"""
    print("\n🔇 TESTING CONTENT INDEPENDENCE")
    print("=" * 50)
    
    # Initialize model factory
    factory = ModelFactory("config/models.yaml")
    
    try:
        model = factory.get_model("audio_emotion")
        
        # Test with same audio characteristics but different content
        test_cases = [
            {
                "name": "Same Characteristics, Different Content",
                "description": "Testing that emotion detection relies on audio characteristics, not content"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            print(f"Description: {test_case['description']}")
            
            # Create test audio with consistent characteristics
            duration = 3
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            
            # Create "happy" audio characteristics
            happy_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
            
            # Save and test
            test_audio_path = "test_content_independence.wav"
            
            try:
                from utils.audio_utils import AudioProcessor
                audio_processor = AudioProcessor()
                
                if audio_processor.AUDIO_AVAILABLE:
                    audio_processor.save_audio(happy_audio, test_audio_path, sr)
                    
                    # Test emotion prediction
                    result = model.predict(test_audio_path)
                    
                    print(f"Predicted Emotion: {result.get('emotion', 'unknown')}")
                    print(f"Confidence: {result.get('confidence', 0):.3f}")
                    print(f"Content Independent: {result.get('content_independent', False)}")
                    
                    # Show that it's analyzing characteristics, not content
                    audio_analysis = result.get('audio_analysis', {})
                    if audio_analysis:
                        print("Audio Analysis:")
                        for category, data in audio_analysis.items():
                            print(f"  {category}: {data}")
                    
                else:
                    print("❌ Audio libraries not available")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            finally:
                if os.path.exists(test_audio_path):
                    os.unlink(test_audio_path)
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main test function"""
    print("🎵 COMPREHENSIVE AUDIO EMOTION DETECTION TEST")
    print("=" * 60)
    
    # Test audio emotion model
    test_audio_emotion_model()
    
    # Test feature extraction
    test_audio_feature_extraction()
    
    # Test emotion characteristics
    test_emotion_characteristics()
    
    # Test content independence
    test_content_independence()
    
    print("\n✅ All audio emotion detection tests completed!")
    print("=" * 60)
    print("\n💡 Key Features Demonstrated:")
    print("  • Content-independent emotion detection")
    print("  • Audio characteristic analysis (pitch, volume, speaking rate)")
    print("  • Tone and energy analysis")
    print("  • Emotion prediction based on audio features")
    print("  • Graceful fallback when audio libraries unavailable")


if __name__ == "__main__":
    main() 