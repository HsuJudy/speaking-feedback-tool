#!/usr/bin/env python3
"""
Demo: MLflow Integration with Speaking Feedback Tool

This script demonstrates how MLflow integrates with the existing
DVC + W&B + Zoom pipeline for model serving and lifecycle management.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from mlflow_integration import MLflowModelManager, MLflowExperimentTracker
from models.model_factory import ModelFactory
from pipeline.inference import InferenceEngine
from utils.wandb_utils import WandbArtifactManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_mlflow_model_registry():
    """Demonstrate MLflow model registry functionality"""
    print("🏗️ MLFLOW MODEL REGISTRY DEMO")
    print("=" * 50)
    
    # Initialize MLflow manager
    model_manager = MLflowModelManager()
    
    # Test with different model types
    factory = ModelFactory("config/models.yaml")
    
    # 1. Log dummy model
    print("\n1. Logging dummy model to MLflow...")
    dummy_model = factory.get_model("dummy")
    dummy_uri = model_manager.log_model(
        model=dummy_model,
        model_name="sentiment-dummy-v1",
        model_type="pyfunc",
        metadata={"model_type": "dummy", "version": "1.0.0", "framework": "custom"}
    )
    
    if dummy_uri:
        print(f"✅ Dummy model logged: {dummy_uri}")
    
    # 2. List registered models
    print("\n2. Listing registered models...")
    models = model_manager.list_models()
    for model in models:
        print(f"  📋 {model['name']} - Versions: {model['latest_versions']}")
    
    # 3. Load and test model
    print("\n3. Loading model from registry...")
    loaded_model = model_manager.load_model("sentiment-dummy-v1")
    if loaded_model:
        test_text = "I love this product!"
        prediction = loaded_model.predict([test_text])
        print(f"✅ Prediction: '{test_text}' → {prediction[0]}")
    
    print("=" * 50)


def demo_mlflow_experiment_tracking():
    """Demonstrate MLflow experiment tracking"""
    print("📊 MLFLOW EXPERIMENT TRACKING DEMO")
    print("=" * 50)
    
    # Initialize experiment tracker
    tracker = MLflowExperimentTracker("speaking-feedback-experiments")
    
    # Start experiment run
    run_id = tracker.start_run("sentiment-analysis-test", {
        "purpose": "demo",
        "model_type": "dummy"
    })
    
    if run_id:
        print(f"✅ Experiment run started: {run_id}")
        
        # Log parameters
        tracker.log_params({
            "model_type": "dummy",
            "confidence_threshold": 0.7,
            "batch_size": 32,
            "test_run": True
        })
        
        # Log metrics
        tracker.log_metrics({
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        })
        
        # Log artifacts (example)
        test_results = {
            "test_samples": 100,
            "positive_predictions": 45,
            "negative_predictions": 35,
            "neutral_predictions": 20
        }
        
        # Save test results to file
        with open("test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        tracker.log_artifact("test_results.json", "test_results")
        print("✅ Test results logged as artifact")
        
        # End run
        tracker.end_run()
        print("✅ Experiment run completed")
    
    print("=" * 50)


def demo_mlflow_model_serving():
    """Demonstrate MLflow model serving"""
    print("🚀 MLFLOW MODEL SERVING DEMO")
    print("=" * 50)
    
    model_manager = MLflowModelManager()
    
    # Check if we have models to serve
    models = model_manager.list_models()
    if not models:
        print("❌ No models available for serving. Please log a model first.")
        return
    
    print(f"📋 Available models for serving: {len(models)}")
    for model in models:
        print(f"  - {model['name']}")
    
    # Note: In a real scenario, you would start the serving server
    # model_manager.serve_model("sentiment-dummy-v1", port=5000)
    print("ℹ️  Model serving would start on port 5000")
    print("ℹ️  Use: mlflow models serve -m models:/sentiment-dummy-v1/latest -p 5000")
    
    print("=" * 50)


def demo_mlflow_with_existing_pipeline():
    """Demonstrate MLflow integration with existing pipeline"""
    print("🔗 MLFLOW + EXISTING PIPELINE INTEGRATION")
    print("=" * 50)
    
    # Initialize components
    model_manager = MLflowModelManager()
    experiment_tracker = MLflowExperimentTracker("pipeline-integration")
    inference_engine = InferenceEngine(model_name="dummy")
    
    # Start experiment run
    run_id = experiment_tracker.start_run("pipeline-test")
    
    if run_id:
        # Test with sample texts
        sample_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        print(f"\n📝 Testing with {len(sample_texts)} sample texts...")
        
        for i, text in enumerate(sample_texts, 1):
            # Use existing inference engine
            result = inference_engine.predict_single(text)
            
            # Log to MLflow
            experiment_tracker.log_metrics({
                f"confidence_{i}": result.get('confidence', 0.0),
                f"sentiment_{i}": 1 if result.get('sentiment') == 'positive' else 0
            })
            
            print(f"  {i}. '{text}' → {result.get('sentiment')} ({result.get('confidence', 0.0):.2f})")
        
        # Log overall statistics
        experiment_tracker.log_metrics({
            "total_samples": len(sample_texts),
            "avg_confidence": 0.75  # Example
        })
        
        experiment_tracker.end_run()
        print("✅ Pipeline integration test completed")
    
    print("=" * 50)


def demo_mlflow_model_lifecycle():
    """Demonstrate MLflow model lifecycle management"""
    print("🔄 MLFLOW MODEL LIFECYCLE DEMO")
    print("=" * 50)
    
    model_manager = MLflowModelManager()
    
    # 1. Development stage
    print("\n1. Development Stage")
    dev_uri = model_manager.log_model(
        model=ModelFactory("config/models.yaml").get_model("dummy"),
        model_name="sentiment-model",
        model_type="pyfunc",
        metadata={"stage": "development", "version": "1.0.0-dev"}
    )
    
    # 2. Staging stage
    print("\n2. Staging Stage")
    if dev_uri:
        # In real scenario, you'd transition the model
        print("ℹ️  Model would be transitioned to Staging")
        # model_manager.transition_model_stage("sentiment-model", "1", "Staging")
    
    # 3. Production stage
    print("\n3. Production Stage")
    print("ℹ️  Model would be transitioned to Production")
    # model_manager.transition_model_stage("sentiment-model", "1", "Production")
    
    # 4. Model serving
    print("\n4. Model Serving")
    print("ℹ️  Model would be served via REST API")
    # model_manager.serve_model("sentiment-model", port=5000)
    
    print("✅ Model lifecycle demonstration completed")
    print("=" * 50)


def main():
    """Run all MLflow demos"""
    print("🎤 MLFLOW INTEGRATION DEMO SUITE")
    print("=" * 60)
    print("This demo shows how MLflow integrates with your existing")
    print("DVC + W&B + Zoom pipeline for complete MLOps workflow.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_mlflow_model_registry()
        demo_mlflow_experiment_tracking()
        demo_mlflow_model_serving()
        demo_mlflow_with_existing_pipeline()
        demo_mlflow_model_lifecycle()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\n📋 Next Steps:")
        print("  1. Install MLflow: pip install mlflow")
        print("  2. Start MLflow UI: mlflow ui")
        print("  3. Serve models: mlflow models serve -m models:/model-name/latest")
        print("  4. Integrate with your Zoom pipeline")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    main() 