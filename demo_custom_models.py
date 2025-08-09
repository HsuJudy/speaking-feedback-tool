#!/usr/bin/env python3
"""
Demo Custom Model Training and Inference
Shows how to train models on dummy data and use them for emotion prediction
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from models.custom_emotion_trainer import CustomEmotionTrainer
from utils.grafana_observability import GrafanaObservability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset():
    """Create dummy dataset for demonstration"""
    print("🎲 Creating dummy dataset...")
    
    # Create dummy features
    np.random.seed(42)
    n_samples = 1000
    
    # Generate dummy audio features
    features = {
        'duration': np.random.uniform(1.0, 10.0, n_samples),
        'sample_rate': np.full(n_samples, 16000),
        'rms_energy': np.random.uniform(0.01, 0.5, n_samples),
        'zero_crossing_rate': np.random.uniform(0.01, 0.1, n_samples),
        'spectral_centroid': np.random.uniform(500, 3000, n_samples),
        'spectral_rolloff': np.random.uniform(1000, 5000, n_samples),
        'mfcc_1': np.random.uniform(-10, 10, n_samples),
        'mfcc_2': np.random.uniform(-10, 10, n_samples),
        'mfcc_3': np.random.uniform(-10, 10, n_samples)
    }
    
    # Create dummy labels (8 emotions)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    labels = np.random.choice(emotions, n_samples, p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['emotion'] = labels
    
    # Save to CSV
    data_dir = Path("data/ravdess")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(data_dir / "features.csv", index=False)
    print(f"✅ Created dummy dataset with {n_samples} samples")
    print(f"Emotion distribution: {df['emotion'].value_counts().to_dict()}")
    
    return df


def demo_model_training():
    """Demo model training with dummy data"""
    print("\n🤖 DEMO: Model Training")
    print("-" * 40)
    
    # Create dummy dataset
    create_dummy_dataset()
    
    # Initialize trainer
    trainer = CustomEmotionTrainer()
    
    # Train all models
    print("Training models...")
    results = trainer.train_all_models()
    
    if not results:
        print("❌ No models were trained successfully")
        return False
    
    # Save models
    saved_models = []
    for model_name, (model, metrics) in results.items():
        model_path = trainer.save_model(model, model_name, metrics)
        saved_models.append(model_path)
        print(f"✅ Saved {model_name} to {model_path}")
    
    # Generate observability report
    report = trainer.generate_observability_report()
    trainer.save_observability_report(report)
    
    print("\n📊 MODEL COMPARISON:")
    print("-" * 30)
    for model_name, data in report["models"].items():
        print(f"{model_name}:")
        print(f"  Accuracy: {data['accuracy']:.3f}")
        print(f"  F1-Score: {data['f1_score']:.3f}")
        print(f"  Training Time: {data['training_time']:.2f}s")
        print(f"  Inference Time: {data['inference_time']:.4f}s")
        print()
    
    print(f"🏆 Best Accuracy: {report['comparison']['best_accuracy']}")
    print(f"🏆 Best F1-Score: {report['comparison']['best_f1_score']}")
    
    return True


def demo_model_inference():
    """Demo model inference with trained models"""
    print("\n🎭 DEMO: Model Inference")
    print("-" * 40)
    
    # Initialize custom emotion trainer
    trainer = CustomEmotionTrainer()
    
    # List available models
    model_files = list(trainer.models_dir.glob("*.pkl"))
    print(f"Available models: {len(model_files)}")
    
    if not model_files:
        print("❌ No trained models found")
        return False
    
    # Test each model
    for model_path in model_files:
        model_name = model_path.stem
        print(f"\nTesting model: {model_name}")
        
        # Load model
        model_package = trainer.load_model(str(model_path))
        
        # Test with dummy features
        test_features = {
            'duration': 3.5,
            'sample_rate': 16000,
            'rms_energy': 0.15,
            'zero_crossing_rate': 0.05,
            'spectral_centroid': 1500,
            'spectral_rolloff': 3000,
            'mfcc_1': 2.5,
            'mfcc_2': -1.2,
            'mfcc_3': 0.8
        }
        
        # Predict emotion
        result = trainer.predict_emotion(model_package, test_features)
        
        print(f"  Test prediction:")
        print(f"    Predicted emotion: {result['emotion']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Probabilities: {result['probabilities']}")
        print()
    
    return True


def demo_observability():
    """Demo observability features"""
    print("\n📊 DEMO: Observability Features")
    print("-" * 40)
    
    # Initialize Grafana observability
    grafana = GrafanaObservability(enabled=True)
    
    # Test metrics logging
    print("Logging model metrics...")
    grafana.log_model_metrics(
        model_name="demo_model",
        accuracy=0.85,
        precision=0.87,
        recall=0.83,
        f1_score=0.85,
        training_time=120.5,
        inference_time=0.002,
        feature_importance={
            "duration": 0.15,
            "rms_energy": 0.25,
            "spectral_centroid": 0.30,
            "mfcc_1": 0.20,
            "mfcc_2": 0.10
        },
        confidence_scores=[0.85, 0.92, 0.78, 0.95, 0.88]
    )
    
    # Test prediction logging
    print("Logging prediction metrics...")
    grafana.log_prediction(
        model_name="demo_model",
        emotion="happy",
        confidence=0.92,
        probabilities={"happy": 0.92, "sad": 0.05, "neutral": 0.03},
        inference_time=0.001,
        features={"duration": 3.5, "rms_energy": 0.15}
    )
    
    # Test pipeline timing
    print("Logging pipeline timing...")
    grafana.log_pipeline_timing(
        pipeline_name="demo_pipeline",
        total_time=2.5,
        stage_timings={
            "audio_loading": 0.5,
            "feature_extraction": 1.2,
            "model_inference": 0.8
        }
    )
    
    print("✅ Observability demo completed")
    return True


def demo_pipeline_integration():
    """Demo pipeline integration"""
    print("\n🔗 DEMO: Pipeline Integration")
    print("-" * 40)
    
    # Initialize custom emotion trainer
    trainer = CustomEmotionTrainer()
    
    # List and load a model
    model_files = list(trainer.models_dir.glob("*.pkl"))
    if not model_files:
        print("❌ No models available for integration test")
        return False
    
    # Load first model
    model_package = trainer.load_model(str(model_files[0]))
    
    # Test pipeline with different feature sets
    test_cases = [
        ("Happy audio", {
            'duration': 3.5, 'sample_rate': 16000, 'rms_energy': 0.25,
            'zero_crossing_rate': 0.08, 'spectral_centroid': 2000,
            'spectral_rolloff': 4000, 'mfcc_1': 5.2, 'mfcc_2': 3.1, 'mfcc_3': 1.8
        }),
        ("Sad audio", {
            'duration': 4.2, 'sample_rate': 16000, 'rms_energy': 0.08,
            'zero_crossing_rate': 0.03, 'spectral_centroid': 800,
            'spectral_rolloff': 2000, 'mfcc_1': -2.1, 'mfcc_2': -1.5, 'mfcc_3': -0.8
        }),
        ("Angry audio", {
            'duration': 2.8, 'sample_rate': 16000, 'rms_energy': 0.35,
            'zero_crossing_rate': 0.12, 'spectral_centroid': 2500,
            'spectral_rolloff': 5000, 'mfcc_1': 8.5, 'mfcc_2': 4.2, 'mfcc_3': 2.1
        })
    ]
    
    for description, features in test_cases:
        print(f"\nTesting: {description}")
        
        # Predict emotion
        result = trainer.predict_emotion(model_package, features)
        print(f"Prediction: {result}")
        
        if "error" not in result:
            print(f"✅ {description} - Success")
        else:
            print(f"❌ {description} - Failed: {result.get('error')}")
    
    return True


def main():
    """Run all demos"""
    print("🎭 CUSTOM MODEL TRAINING DEMO")
    print("=" * 50)
    
    demos = [
        ("Model Training", demo_model_training),
        ("Model Inference", demo_model_inference),
        ("Observability", demo_observability),
        ("Pipeline Integration", demo_pipeline_integration)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*60}")
            result = demo_func()
            results[demo_name] = result
        except Exception as e:
            print(f"❌ {demo_name} failed with exception: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n📋 DEMO SUMMARY")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for demo_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{demo_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("🎉 All demos passed! Custom model system is working correctly.")
        print("\n🎯 NEXT STEPS:")
        print("1. Download real RAVDESS dataset: python data/ravdess_downloader.py")
        print("2. Train models on real data: python train_custom_models.py")
        print("3. Use models in production: from models.custom_emotion_trainer import CustomEmotionTrainer")
        print("4. Monitor with Grafana: Setup Grafana dashboard for real-time monitoring")
    else:
        print("⚠️  Some demos failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    main() 