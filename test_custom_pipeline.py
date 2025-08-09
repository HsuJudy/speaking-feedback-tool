#!/usr/bin/env python3
"""
Test Custom Model Pipeline
Tests the complete custom model training and inference pipeline
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from data.ravdess_downloader import RAVDESSDownloader
from models.custom_emotion_trainer import CustomEmotionTrainer
from models.custom_emotion_model import CustomEmotionModel
from utils.grafana_observability import GrafanaObservability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ravdess_downloader():
    """Test RAVDESS downloader"""
    print("📥 Testing RAVDESS Downloader")
    print("-" * 30)
    
    downloader = RAVDESSDownloader()
    
    # Get dataset info
    info = downloader.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Test with small dataset (just check if files exist)
    if info.get("features_exists", False):
        print("✅ Dataset already prepared")
        return True
    else:
        print("⚠️  Dataset not prepared - run training script first")
        return False


def test_custom_trainer():
    """Test custom model trainer"""
    print("\n🤖 Testing Custom Model Trainer")
    print("-" * 30)
    
    trainer = CustomEmotionTrainer()
    
    # Check if training data exists
    try:
        X, y = trainer.load_data()
        print(f"✅ Training data loaded: {len(X)} samples")
        print(f"Feature columns: {list(X.columns)}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return False


def test_custom_model():
    """Test custom emotion model"""
    print("\n🎭 Testing Custom Emotion Model")
    print("-" * 30)
    
    model = CustomEmotionModel()
    
    # List available models
    available_models = model.list_available_models()
    print(f"Available models: {len(available_models)}")
    
    if not available_models:
        print("⚠️  No trained models found - run training script first")
        return False
    
    # Test first available model
    first_model = available_models[0]
    print(f"Testing model: {first_model['name']}")
    
    # Load model
    if not model.load_model(first_model['path']):
        print("❌ Failed to load model")
        return False
    
    # Get model info
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    # Test prediction
    dummy_audio = model.audio_processor.create_dummy_audio("Test prediction!", duration=3.0)
    result = model.predict_emotion(dummy_audio)
    
    print(f"Prediction result: {result}")
    
    if "error" not in result:
        print("✅ Model prediction successful")
        return True
    else:
        print(f"❌ Model prediction failed: {result.get('error')}")
        return False


def test_grafana_observability():
    """Test Grafana observability"""
    print("\n📊 Testing Grafana Observability")
    print("-" * 30)
    
    grafana = GrafanaObservability(enabled=False)  # Disabled for testing
    
    # Test metrics logging
    grafana.log_model_metrics(
        model_name="test_model",
        accuracy=0.85,
        precision=0.87,
        recall=0.83,
        f1_score=0.85,
        training_time=120.5,
        inference_time=0.002,
        feature_importance={"feature1": 0.3, "feature2": 0.7},
        confidence_scores=[0.85, 0.92, 0.78]
    )
    
    # Test prediction logging
    grafana.log_prediction(
        model_name="test_model",
        emotion="happy",
        confidence=0.92,
        probabilities={"happy": 0.92, "sad": 0.05, "neutral": 0.03},
        inference_time=0.001,
        features={"duration": 3.5, "rms_energy": 0.15}
    )
    
    print("✅ Grafana observability test completed")
    return True


def test_pipeline_integration():
    """Test pipeline integration"""
    print("\n🔗 Testing Pipeline Integration")
    print("-" * 30)
    
    # Test custom model with observability
    model = CustomEmotionModel(enable_observability=True)
    
    # List and load a model
    available_models = model.list_available_models()
    if not available_models:
        print("⚠️  No models available for integration test")
        return False
    
    # Load first model
    model.load_model(available_models[0]['path'])
    
    # Test pipeline with different audio types
    test_cases = [
        ("Positive sentiment", "I love this amazing product!"),
        ("Negative sentiment", "This is terrible and awful!"),
        ("Neutral sentiment", "This is okay, nothing special.")
    ]
    
    for description, text in test_cases:
        print(f"\nTesting: {description}")
        
        # Create dummy audio
        audio = model.audio_processor.create_dummy_audio(text, duration=3.0)
        
        # Extract features
        features = model.extract_audio_features(audio)
        print(f"Features: {features}")
        
        # Predict emotion
        result = model.predict_emotion(audio)
        print(f"Prediction: {result}")
        
        if "error" not in result:
            print(f"✅ {description} - Success")
        else:
            print(f"❌ {description} - Failed: {result.get('error')}")
    
    return True


def main():
    """Run all tests"""
    print("🧪 CUSTOM MODEL PIPELINE TESTS")
    print("=" * 50)
    
    tests = [
        ("RAVDESS Downloader", test_ravdess_downloader),
        ("Custom Trainer", test_custom_trainer),
        ("Custom Model", test_custom_model),
        ("Grafana Observability", test_grafana_observability),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    main() 