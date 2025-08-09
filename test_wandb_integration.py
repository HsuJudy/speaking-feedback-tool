"""
Test script for W&B integration
Demonstrates artifact tracking, experiment logging, and model versioning
"""

import json
import os
from models.model_factory import ModelFactory
from pipeline.inference import InferenceEngine
from utils.wandb_utils import WandbArtifactManager


def test_wandb_integration():
    """Test W&B integration with the sentiment analysis pipeline"""
    print("🔮 TESTING W&B INTEGRATION")
    print("=" * 50)
    
    # Initialize W&B artifact manager
    wandb_manager = WandbArtifactManager("vibe-check-wandb-test")
    
    # Test run initialization
    print("\n📊 Initializing W&B run...")
    success = wandb_manager.init_run("sentiment-analysis-test", {
        "project": "vibe-check",
        "framework": "modular-mlops",
        "version": "v1.0",
        "models": ["dummy", "huggingface", "nemo", "video_sentiment"]
    })
    
    if not success:
        print("❌ Failed to initialize W&B run")
        return
    
    # Test model factory with W&B
    print("\n🏭 Testing ModelFactory with W&B...")
    factory = ModelFactory("config/models.yaml", use_wandb=True)
    
    # Test different models
    test_models = ["dummy", "huggingface", "nemo", "video_sentiment"]
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it!",
        "The movie was okay, nothing special."
    ]
    
    for model_name in test_models:
        print(f"\n--- Testing {model_name} model ---")
        
        try:
            # Get model with W&B tracking
            model = factory.get_model(model_name)
            print(f"✅ Model loaded: {model.get_model_info()}")
            
            # Test single prediction
            result = model.predict(test_texts[0])
            print(f"Single prediction: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            
            # Test batch prediction
            batch_results = model.batch_predict(test_texts)
            print(f"Batch predictions: {len(batch_results)} results")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # Test inference engine with W&B
    print("\n🔍 Testing InferenceEngine with W&B...")
    
    for model_name in ["dummy", "video_sentiment"]:
        print(f"\n--- Testing {model_name} inference ---")
        
        try:
            # Initialize inference engine with W&B
            inference_engine = InferenceEngine(model_name=model_name, use_wandb=True)
            
            # Test single prediction
            result = inference_engine.predict_single(test_texts[0])
            print(f"Single inference: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            
            # Test batch prediction
            batch_results = inference_engine.predict_batch(test_texts)
            print(f"Batch inference: {len(batch_results)} results")
            
        except Exception as e:
            print(f"❌ Error with {model_name} inference: {e}")
    
    # Test artifact logging
    print("\n📦 Testing artifact logging...")
    
    # Log dataset artifact
    dataset_artifact = wandb_manager.log_dataset_artifact(
        "data/sample_inputs.json",
        "sentiment-test-dataset",
        "sentiment"
    )
    print(f"Dataset artifact: {dataset_artifact}")
    
    # Log video artifact
    video_artifact = wandb_manager.log_video_artifact(
        "test_video.mp4",
        "test-video-sentiment",
        "sentiment"
    )
    print(f"Video artifact: {video_artifact}")
    
    # Log metrics
    print("\n📈 Logging metrics...")
    metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "total_predictions": 15,
        "models_tested": len(test_models)
    }
    wandb_manager.log_metrics(metrics)
    
    # Log predictions
    print("\n🔮 Logging predictions...")
    predictions = [
        {"text": "I love this!", "sentiment": "positive", "confidence": 0.9, "model": "dummy"},
        {"text": "This is terrible!", "sentiment": "negative", "confidence": 0.8, "model": "dummy"},
        {"text": "It's okay.", "sentiment": "neutral", "confidence": 0.6, "model": "dummy"}
    ]
    wandb_manager.log_predictions(predictions, "dummy-model")
    
    # Finish W&B run
    wandb_manager.finish_run()
    
    print("\n✅ W&B integration testing completed!")


def test_wandb_artifacts():
    """Test W&B artifact functionality"""
    print("\n🎬 TESTING W&B ARTIFACTS")
    print("=" * 50)
    
    # Initialize artifact manager
    manager = WandbArtifactManager("vibe-check-artifacts")
    
    # Test run initialization
    success = manager.init_run("artifact-test", {
        "artifact_types": ["model", "dataset", "video"],
        "test_mode": True
    })
    
    if success:
        # Test model artifacts
        print("\n🤖 Testing model artifacts...")
        model_artifacts = [
            ("dummy-sentiment", "sentiment", {"accuracy": 0.85}),
            ("huggingface-sentiment", "sentiment", {"accuracy": 0.92}),
            ("nemo-speech", "speech", {"accuracy": 0.88})
        ]
        
        for model_name, model_type, metadata in model_artifacts:
            artifact = manager.log_model_artifact(
                f"models/{model_name}_model.json",
                model_name,
                model_type,
                metadata
            )
            print(f"  {model_name}: {artifact}")
        
        # Test dataset artifacts
        print("\n📚 Testing dataset artifacts...")
        dataset_artifacts = [
            ("sentiment-dataset", "sentiment"),
            ("video-dataset", "video"),
            ("audio-dataset", "audio")
        ]
        
        for dataset_name, dataset_type in dataset_artifacts:
            artifact = manager.log_dataset_artifact(
                f"data/{dataset_name}.json",
                dataset_name,
                dataset_type
            )
            print(f"  {dataset_name}: {artifact}")
        
        # Test video artifacts
        print("\n🎬 Testing video artifacts...")
        video_artifacts = [
            ("test-video-1", "sentiment"),
            ("test-video-2", "speech"),
            ("test-video-3", "emotion")
        ]
        
        for video_name, video_type in video_artifacts:
            artifact = manager.log_video_artifact(
                f"videos/{video_name}.mp4",
                video_name,
                video_type
            )
            print(f"  {video_name}: {artifact}")
        
        # Finish run
        manager.finish_run()
        
        print("\n✅ W&B artifact testing completed!")
    else:
        print("❌ Failed to initialize W&B run for artifacts")


def test_wandb_experiment_tracking():
    """Test W&B experiment tracking capabilities"""
    print("\n🧪 TESTING W&B EXPERIMENT TRACKING")
    print("=" * 50)
    
    # Initialize artifact manager
    manager = WandbArtifactManager("vibe-check-experiments")
    
    # Test run initialization
    success = manager.init_run("experiment-tracking-test", {
        "experiment_type": "sentiment_analysis",
        "models": ["dummy", "huggingface", "nemo"],
        "metrics": ["accuracy", "precision", "recall", "f1"]
    })
    
    if success:
        # Simulate experiment iterations
        for epoch in range(5):
            print(f"\n📊 Epoch {epoch + 1}/5")
            
            # Simulate training metrics
            training_metrics = {
                "epoch": epoch + 1,
                "loss": 0.1 + (epoch * 0.02),
                "accuracy": 0.8 + (epoch * 0.03),
                "precision": 0.75 + (epoch * 0.02),
                "recall": 0.82 + (epoch * 0.01)
            }
            
            manager.log_metrics(training_metrics, step=epoch)
            print(f"  Logged metrics: {list(training_metrics.keys())}")
            
            # Simulate validation predictions
            if epoch % 2 == 0:  # Every other epoch
                predictions = [
                    {"text": f"Sample {i}", "sentiment": "positive", "confidence": 0.8 + (i * 0.05)}
                    for i in range(3)
                ]
                manager.log_predictions(predictions, f"model-epoch-{epoch}")
                print(f"  Logged {len(predictions)} predictions")
        
        # Finish run
        manager.finish_run()
        
        print("\n✅ W&B experiment tracking completed!")
    else:
        print("❌ Failed to initialize W&B run for experiments")


def main():
    """Main test function"""
    print("🔮 COMPREHENSIVE W&B INTEGRATION TEST")
    print("=" * 60)
    
    # Test basic integration
    test_wandb_integration()
    
    # Test artifacts
    test_wandb_artifacts()
    
    # Test experiment tracking
    test_wandb_experiment_tracking()
    
    print("\n✅ All W&B integration tests completed!")
    print("=" * 60)
    print("\n💡 Key Features Demonstrated:")
    print("  • Model artifact tracking and versioning")
    print("  • Dataset artifact management")
    print("  • Video artifact logging")
    print("  • Experiment metrics tracking")
    print("  • Prediction logging and visualization")
    print("  • Integration with existing pipeline")


if __name__ == "__main__":
    main() 