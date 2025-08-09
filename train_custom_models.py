#!/usr/bin/env python3
"""
Comprehensive Custom Model Training Script
Downloads RAVDESS data, trains models, and integrates with observability
"""

import os
import sys
import time
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    print("🎭 CUSTOM EMOTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Download and prepare RAVDESS dataset
    print("\n📥 STEP 1: Downloading RAVDESS Dataset")
    print("-" * 40)
    
    downloader = RAVDESSDownloader()
    
    # Check if data already exists
    dataset_info = downloader.get_dataset_info()
    print(f"Dataset info: {dataset_info}")
    
    if not dataset_info.get("features_exists", False):
        print("Downloading and preparing RAVDESS dataset...")
        
        # Download dataset
        if not downloader.download_dataset():
            print("❌ Failed to download dataset")
            return
        
        # Extract dataset
        if not downloader.extract_dataset():
            print("❌ Failed to extract dataset")
            return
        
        # Prepare training data
        features_df, labels_df = downloader.prepare_training_data()
        
        if features_df.empty:
            print("❌ Failed to prepare training data")
            return
        
        print(f"✅ Dataset prepared: {len(features_df)} samples")
    else:
        print("✅ Dataset already prepared")
    
    # Step 2: Train custom models
    print("\n🤖 STEP 2: Training Custom Models")
    print("-" * 40)
    
    trainer = CustomEmotionTrainer()
    
    # Train all models
    results = trainer.train_all_models()
    
    if not results:
        print("❌ No models were trained successfully")
        return
    
    # Save models and generate observability report
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
    print(f"⚡ Fastest Training: {report['comparison']['fastest_training']}")
    print(f"⚡ Fastest Inference: {report['comparison']['fastest_inference']}")
    
    # Step 3: Test custom models
    print("\n🧪 STEP 3: Testing Custom Models")
    print("-" * 40)
    
    # Initialize custom emotion model
    custom_model = CustomEmotionModel()
    
    # List available models
    available_models = custom_model.list_available_models()
    print(f"Available trained models: {len(available_models)}")
    
    for model_info in available_models:
        print(f"  - {model_info['name']}: {model_info['path']}")
    
    # Test each model
    for model_info in available_models:
        print(f"\nTesting model: {model_info['name']}")
        
        # Load model
        custom_model.load_model(model_info['path'])
        
        # Get model info
        info = custom_model.get_model_info()
        print(f"Model info: {info}")
        
        # Test with dummy audio
        dummy_audio = custom_model.audio_processor.create_dummy_audio(
            "I love this product! It's amazing!", 
            duration=3.0
        )
        
        result = custom_model.predict_emotion(dummy_audio)
        print(f"Prediction result: {result}")
        
        # Test with negative sentiment
        negative_audio = custom_model.audio_processor.create_dummy_audio(
            "This is terrible! I hate it!", 
            duration=3.0
        )
        
        negative_result = custom_model.predict_emotion(negative_audio)
        print(f"Negative prediction: {negative_result}")
    
    # Step 4: Setup Grafana observability
    print("\n📊 STEP 4: Setting up Grafana Observability")
    print("-" * 40)
    
    grafana = GrafanaObservability()
    
    # Check Grafana connectivity
    if grafana.health_check():
        print("✅ Grafana is accessible")
        
        # Setup dashboard
        if grafana.setup_grafana_dashboard():
            print("✅ Grafana dashboard created")
        else:
            print("⚠️  Could not create Grafana dashboard")
    else:
        print("⚠️  Grafana is not accessible. Observability will be disabled.")
    
    # Step 5: Integration with existing pipeline
    print("\n🔗 STEP 5: Pipeline Integration")
    print("-" * 40)
    
    # Test integration with existing audio processor
    audio_processor = custom_model.audio_processor
    
    # Create test audio
    test_audio = audio_processor.create_dummy_audio("Testing the pipeline integration!", duration=5.0)
    
    # Extract features
    features = custom_model.extract_audio_features(test_audio)
    print(f"Extracted features: {features}")
    
    # Test prediction with observability
    prediction_result = custom_model.predict_emotion(test_audio)
    print(f"Pipeline prediction: {prediction_result}")
    
    # Step 6: Summary and next steps
    print("\n📋 SUMMARY")
    print("-" * 40)
    print(f"✅ Downloaded and prepared RAVDESS dataset")
    print(f"✅ Trained {len(results)} custom models:")
    for model_name in results.keys():
        print(f"   - {model_name}")
    print(f"✅ Saved models to: {trainer.models_dir}")
    print(f"✅ Generated observability report")
    print(f"✅ Tested model integration")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Use trained models in your pipeline:")
    print("   from models.custom_emotion_model import CustomEmotionModel")
    print("   model = CustomEmotionModel('models/custom/random_forest_emotion_model.pkl')")
    print("   result = model.predict_emotion_from_file('audio.wav')")
    
    print("\n2. Monitor performance with Grafana:")
    print("   - Access Grafana at http://localhost:3000")
    print("   - View ML pipeline metrics and SHAP values")
    
    print("\n3. Integrate with existing pipeline:")
    print("   - Models are saved as .pkl files")
    print("   - Can be loaded and used in production")
    print("   - Observability metrics are automatically logged")
    
    print("\n🚀 Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 