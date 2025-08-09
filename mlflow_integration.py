"""
MLflow Integration for Speaking Feedback Tool

This module provides MLflow integration for model serving, experiment tracking,
and model lifecycle management. It works alongside the existing DVC and W&B setup.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import shutil

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install mlflow for model serving.")

from models.model_factory import ModelFactory
from utils.wandb_utils import WandbArtifactManager

logger = logging.getLogger(__name__)


class MLflowModelManager:
    """Manages MLflow model registry and serving"""
    
    def __init__(self, tracking_uri: str = None, registry_uri: str = None):
        """Initialize MLflow manager"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Model serving disabled.")
            return
            
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
        self.registry_uri = registry_uri or os.getenv('MLFLOW_REGISTRY_URI', 'sqlite:///mlflow.db')
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn", 
                  artifacts: Dict[str, str] = None, metadata: Dict[str, Any] = None) -> str:
        """Log a model to MLflow registry"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot log model: MLflow not available")
            return None
            
        try:
            with mlflow.start_run():
                # Log model based on type
                if model_type == "sklearn":
                    mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
                elif model_type == "pytorch":
                    mlflow.pytorch.log_model(model, model_name, registered_model_name=model_name)
                elif model_type == "pyfunc":
                    mlflow.pyfunc.log_model(model_name, python_model=model, registered_model_name=model_name)
                
                # Log artifacts if provided
                if artifacts:
                    for artifact_name, artifact_path in artifacts.items():
                        mlflow.log_artifact(artifact_path, artifact_path=artifact_name)
                
                # Log metadata
                if metadata:
                    mlflow.log_params(metadata)
                
                # Get model URI
                model_uri = f"models:/{model_name}/latest"
                logger.info(f"Model logged successfully: {model_uri}")
                return model_uri
                
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
            return None
    
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a model from MLflow registry"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot load model: MLflow not available")
            return None
            
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded successfully: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            return None
    
    def serve_model(self, model_name: str, port: int = 5000) -> None:
        """Serve a model via MLflow REST API"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot serve model: MLflow not available")
            return
            
        try:
            model_uri = f"models:/{model_name}/latest"
            mlflow.pyfunc.serve(model_uri, port=port)
            logger.info(f"Model serving started on port {port}")
        except Exception as e:
            logger.error(f"Error serving model: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the registry"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot list models: MLflow not available")
            return []
            
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            return [
                {
                    "name": model.name,
                    "latest_versions": [v.version for v in model.latest_versions],
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """Transition a model to a specific stage (Staging, Production, etc.)"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot transition model: MLflow not available")
            return False
            
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(model_name, version, stage)
            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
            return True
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False


class MLflowExperimentTracker:
    """Tracks experiments with MLflow"""
    
    def __init__(self, experiment_name: str = "speaking-feedback"):
        """Initialize experiment tracker"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Experiment tracking disabled.")
            return
            
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start an MLflow run"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot start run: MLflow not available")
            return None
            
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"MLflow run started: {run.info.run_id}")
            return run.info.run_id
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot log metrics: MLflow not available")
            return
            
        try:
            mlflow.log_metrics(metrics)
            logger.info(f"Metrics logged: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot log params: MLflow not available")
            return
            
        try:
            mlflow.log_params(params)
            logger.info(f"Parameters logged: {list(params.keys())}")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log an artifact to MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot log artifact: MLflow not available")
            return
            
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Artifact logged: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        if not MLFLOW_AVAILABLE:
            logger.warning("Cannot end run: MLflow not available")
            return
            
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")


def integrate_with_existing_pipeline():
    """Demonstrate MLflow integration with existing pipeline"""
    print("🔧 MLFLOW INTEGRATION WITH EXISTING PIPELINE")
    print("=" * 50)
    
    # Initialize MLflow components
    model_manager = MLflowModelManager()
    experiment_tracker = MLflowExperimentTracker("speaking-feedback-v1")
    
    # Test with existing model factory
    try:
        factory = ModelFactory("config/models.yaml")
        model = factory.get_model("dummy")
        
        # Log model to MLflow
        model_uri = model_manager.log_model(
            model=model,
            model_name="sentiment-analysis-dummy",
            model_type="pyfunc",
            metadata={"model_type": "dummy", "version": "1.0.0"}
        )
        
        if model_uri:
            print(f"✅ Model logged to MLflow: {model_uri}")
            
            # List all models
            models = model_manager.list_models()
            print(f"📋 Registered models: {len(models)}")
            for model_info in models:
                print(f"  - {model_info['name']}")
        
        # Test experiment tracking
        run_id = experiment_tracker.start_run("sentiment-test")
        if run_id:
            experiment_tracker.log_metrics({"accuracy": 0.85, "precision": 0.82})
            experiment_tracker.log_params({"model_type": "dummy", "test_run": True})
            experiment_tracker.end_run()
            print(f"✅ Experiment tracked: {run_id}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print("=" * 50)


def create_mlflow_serving_endpoint():
    """Create a Flask endpoint that serves MLflow models"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    model_manager = MLflowModelManager()
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Make predictions using MLflow models"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            model_name = data.get('model', 'sentiment-analysis-dummy')
            
            # Load model from MLflow
            model = model_manager.load_model(model_name)
            if model:
                # Make prediction
                prediction = model.predict([text])
                return jsonify({
                    "text": text,
                    "prediction": prediction[0],
                    "model": model_name
                })
            else:
                return jsonify({"error": "Model not found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/models', methods=['GET'])
    def list_models():
        """List available models"""
        models = model_manager.list_models()
        return jsonify({"models": models})
    
    return app


if __name__ == "__main__":
    # Test MLflow integration
    integrate_with_existing_pipeline()
    
    # Create serving endpoint
    app = create_mlflow_serving_endpoint()
    print("🚀 MLflow serving endpoint created")
    print("   POST /predict - Make predictions")
    print("   GET  /models  - List models") 