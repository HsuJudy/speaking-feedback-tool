"""
Weights & Biases (W&B) utilities for artifact tracking
Handles model artifacts, datasets, and experiment logging
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional, List
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available. Install wandb for experiment tracking.")


class WandbArtifactManager:
    """Manages W&B artifacts for model tracking and versioning"""
    
    def __init__(self, project_name: str = "vibe-check-sentiment", entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.artifact_cache = {}
        
        if not WANDB_AVAILABLE:
            print("[WandbArtifactManager] Warning: W&B not available")
    
    def init_run(self, run_name: str = None, config: Dict[str, Any] = None):
        """Initialize a W&B run"""
        if not WANDB_AVAILABLE:
            print("[WandbArtifactManager] Cannot init run: W&B not available")
            return False
        
        try:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config or {}
            )
            print(f"[WandbArtifactManager] Initialized run: {self.run.name}")
            return True
        except Exception as e:
            print(f"[WandbArtifactManager] Error initializing run: {e}")
            return False
    
    def log_model_artifact(self, model_path: str, model_name: str, model_type: str, 
                          metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Log a model as a W&B artifact
        
        Args:
            model_path (str): Path to model file
            model_name (str): Name of the model
            model_type (str): Type of model (e.g., 'sentiment', 'speech')
            metadata (dict): Additional metadata
            
        Returns:
            str: Artifact version name
        """
        if not WANDB_AVAILABLE or not self.run:
            print("[WandbArtifactManager] Cannot log artifact: W&B not available or run not initialized")
            return None
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=f"{model_name}-{model_type}",
                type="model",
                description=f"{model_type} model for sentiment analysis"
            )
            
            # Add model file
            if os.path.exists(model_path):
                artifact.add_file(model_path)
            else:
                print(f"[WandbArtifactManager] Warning: Model file {model_path} not found")
                # Create dummy model file for demonstration
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump({
                        "model_type": model_type,
                        "model_name": model_name,
                        "version": "v1.0",
                        "dummy": True
                    }, f)
                    artifact.add_file(f.name)
                    os.unlink(f.name)
            
            # Add metadata
            if metadata:
                artifact.metadata.update(metadata)
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            artifact_name = f"{model_name}-{model_type}:v{artifact.version}"
            print(f"[WandbArtifactManager] Logged model artifact: {artifact_name}")
            
            # Cache artifact
            self.artifact_cache[model_name] = artifact_name
            
            return artifact_name
            
        except Exception as e:
            print(f"[WandbArtifactManager] Error logging model artifact: {e}")
            return None
    
    def log_dataset_artifact(self, dataset_path: str, dataset_name: str, 
                           dataset_type: str = "sentiment") -> Optional[str]:
        """
        Log a dataset as a W&B artifact
        
        Args:
            dataset_path (str): Path to dataset file
            dataset_name (str): Name of the dataset
            dataset_type (str): Type of dataset
            
        Returns:
            str: Artifact version name
        """
        if not WANDB_AVAILABLE or not self.run:
            print("[WandbArtifactManager] Cannot log dataset: W&B not available or run not initialized")
            return None
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=f"{dataset_name}-{dataset_type}",
                type="dataset",
                description=f"{dataset_type} dataset for sentiment analysis"
            )
            
            # Add dataset file
            if os.path.exists(dataset_path):
                artifact.add_file(dataset_path)
            else:
                print(f"[WandbArtifactManager] Warning: Dataset file {dataset_path} not found")
                # Create dummy dataset for demonstration
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump({
                        "dataset_type": dataset_type,
                        "dataset_name": dataset_name,
                        "samples": [
                            {"text": "I love this product!", "sentiment": "positive"},
                            {"text": "This is terrible!", "sentiment": "negative"},
                            {"text": "It's okay.", "sentiment": "neutral"}
                        ]
                    }, f)
                    artifact.add_file(f.name)
                    os.unlink(f.name)
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            artifact_name = f"{dataset_name}-{dataset_type}:v{artifact.version}"
            print(f"[WandbArtifactManager] Logged dataset artifact: {artifact_name}")
            
            return artifact_name
            
        except Exception as e:
            print(f"[WandbArtifactManager] Error logging dataset artifact: {e}")
            return None
    
    def log_video_artifact(self, video_path: str, video_name: str, 
                          video_type: str = "sentiment") -> Optional[str]:
        """
        Log a video as a W&B artifact
        
        Args:
            video_path (str): Path to video file
            video_name (str): Name of the video
            video_type (str): Type of video content
            
        Returns:
            str: Artifact version name
        """
        if not WANDB_AVAILABLE or not self.run:
            print("[WandbArtifactManager] Cannot log video: W&B not available or run not initialized")
            return None
        
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=f"{video_name}-{video_type}",
                type="video",
                description=f"{video_type} video for sentiment analysis"
            )
            
            # Add video file
            if os.path.exists(video_path):
                artifact.add_file(video_path)
            else:
                print(f"[WandbArtifactManager] Warning: Video file {video_path} not found")
                # Create dummy video info for demonstration
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump({
                        "video_type": video_type,
                        "video_name": video_name,
                        "duration": 5.0,
                        "resolution": "640x480",
                        "dummy": True
                    }, f)
                    artifact.add_file(f.name)
                    os.unlink(f.name)
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            artifact_name = f"{video_name}-{video_type}:v{artifact.version}"
            print(f"[WandbArtifactManager] Logged video artifact: {artifact_name}")
            
            return artifact_name
            
        except Exception as e:
            print(f"[WandbArtifactManager] Error logging video artifact: {e}")
            return None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B"""
        if not WANDB_AVAILABLE or not self.run:
            print("[WandbArtifactManager] Cannot log metrics: W&B not available or run not initialized")
            return
        
        try:
            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
            print(f"[WandbArtifactManager] Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            print(f"[WandbArtifactManager] Error logging metrics: {e}")
    
    def log_predictions(self, predictions: List[Dict[str, Any]], model_name: str):
        """Log predictions as a table"""
        if not WANDB_AVAILABLE or not self.run:
            print("[WandbArtifactManager] Cannot log predictions: W&B not available or run not initialized")
            return
        
        try:
            # Create table
            table = wandb.Table(columns=["text", "sentiment", "confidence", "model"])
            
            for pred in predictions:
                table.add_data(
                    pred.get("text", ""),
                    pred.get("sentiment", ""),
                    pred.get("confidence", 0.0),
                    model_name
                )
            
            self.run.log({f"{model_name}_predictions": table})
            print(f"[WandbArtifactManager] Logged {len(predictions)} predictions for {model_name}")
            
        except Exception as e:
            print(f"[WandbArtifactManager] Error logging predictions: {e}")
    
    def get_model_artifact(self, model_name: str, model_type: str, version: str = "latest"):
        """
        Get a model artifact from W&B
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model
            version (str): Version to get (default: latest)
            
        Returns:
            wandb.Artifact: Model artifact
        """
        if not WANDB_AVAILABLE:
            print("[WandbArtifactManager] Cannot get artifact: W&B not available")
            return None
        
        try:
            artifact_name = f"{model_name}-{model_type}:{version}"
            artifact = wandb.use_artifact(artifact_name)
            print(f"[WandbArtifactManager] Retrieved model artifact: {artifact_name}")
            return artifact
        except Exception as e:
            print(f"[WandbArtifactManager] Error getting model artifact: {e}")
            return None
    
    def finish_run(self):
        """Finish the current W&B run"""
        if self.run:
            self.run.finish()
            print("[WandbArtifactManager] Finished W&B run")


def test_wandb_artifacts():
    """Test W&B artifact functionality"""
    print("🔮 TESTING W&B ARTIFACTS")
    print("=" * 50)
    
    # Initialize artifact manager
    manager = WandbArtifactManager("vibe-check-test")
    
    # Test run initialization
    print("\n📊 Initializing W&B run...")
    success = manager.init_run("test-run", {
        "model_type": "sentiment",
        "framework": "dummy",
        "version": "v1.0"
    })
    
    if success:
        # Test model artifact logging
        print("\n🤖 Logging model artifacts...")
        model_artifact = manager.log_model_artifact(
            "models/dummy_model.json",
            "dummy-sentiment",
            "sentiment",
            {"accuracy": 0.85, "framework": "dummy"}
        )
        
        # Test dataset artifact logging
        print("\n📚 Logging dataset artifacts...")
        dataset_artifact = manager.log_dataset_artifact(
            "data/sample_inputs.json",
            "sentiment-dataset",
            "sentiment"
        )
        
        # Test video artifact logging
        print("\n🎬 Logging video artifacts...")
        video_artifact = manager.log_video_artifact(
            "test_video.mp4",
            "test-video",
            "sentiment"
        )
        
        # Test metrics logging
        print("\n📈 Logging metrics...")
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        manager.log_metrics(metrics)
        
        # Test predictions logging
        print("\n🔮 Logging predictions...")
        predictions = [
            {"text": "I love this!", "sentiment": "positive", "confidence": 0.9},
            {"text": "This is terrible!", "sentiment": "negative", "confidence": 0.8},
            {"text": "It's okay.", "sentiment": "neutral", "confidence": 0.6}
        ]
        manager.log_predictions(predictions, "dummy-model")
        
        # Finish run
        manager.finish_run()
        
        print("\n✅ W&B artifact testing completed!")
    else:
        print("❌ Failed to initialize W&B run")


if __name__ == "__main__":
    test_wandb_artifacts() 