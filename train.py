"""
Training Pipeline for MLOps Learning
Learn: model versioning, experiment tracking, hyperparameter management, reproducible training
"""

import json
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from data.version_control import DataVersionControl
from utils.audio_features import AudioFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training pipeline for MLOps learning
    
    Concepts covered:
    - Reproducible training with fixed seeds
    - Model versioning and artifact management
    - Hyperparameter management
    - Experiment tracking
    - Model evaluation and validation
    """
    
    def __init__(self, 
                 model_dir: str = "models",
                 data_dir: str = "data",
                 experiment_name: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path(data_dir)
        self.data_version_control = DataVersionControl(data_dir)
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.model_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Training components
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Model registry
        self.model_registry_file = self.model_dir / "model_registry.json"
        self.model_registry = self._load_model_registry()
        
        logger.info(f"ModelTrainer initialized for experiment: {self.experiment_name}")
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry for versioning"""
        if self.model_registry_file.exists():
            try:
                with open(self.model_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                return {}
        return {}
    
    def _save_model_registry(self):
        """Save model registry"""
        try:
            with open(self.model_registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def prepare_training_data(self, 
                             dataset_path: str,
                             target_column: str = "emotion",
                             test_size: float = 0.2,
                             random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with reproducible splits
        
        MLOps concept: Reproducible data preparation
        - Same data + same seed = same splits
        - Enables experiment reproducibility
        """
        logger.info(f"Preparing training data from {dataset_path}")
        
        # Load data
        if dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            data = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Create reproducible train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        # Scale features (important for some models)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   model_type: str = "random_forest",
                   hyperparameters: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model with hyperparameters
        
        MLOps concept: Hyperparameter management
        - Track all hyperparameters for reproducibility
        - Enable hyperparameter tuning
        - Version control for model configurations
        """
        logger.info(f"Training {model_type} model...")
        
        # Set default hyperparameters
        default_params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42
            }
        }
        
        # Use provided hyperparameters or defaults
        params = hyperparameters or default_params.get(model_type, {})
        
        # Initialize model
        if model_type == "random_forest":
            model = RandomForestClassifier(**params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**params)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        training_info = {
            "model_type": model_type,
            "hyperparameters": params,
            "training_time": training_time,
            "cv_mean_score": cv_scores.mean(),
            "cv_std_score": cv_scores.std(),
            "feature_names": [f"feature_{i}" for i in range(X_train.shape[1])]
        }
        
        logger.info(f"Model trained in {training_time:.3f}s, CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model, training_info
    
    def evaluate_model(self,
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        MLOps concept: Comprehensive model evaluation
        - Multiple metrics for different aspects
        - Detailed performance analysis
        - Error analysis for debugging
        """
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(model_info["feature_names"], model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(model_info["feature_names"], model.coef_[0]))
        
        evaluation_results = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "feature_importance": feature_importance,
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba.tolist() if y_pred_proba is not None else None,
            "test_samples": len(X_test)
        }
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        
        return evaluation_results
    
    def save_model(self,
                   model: Any,
                   model_info: Dict[str, Any],
                   evaluation_results: Dict[str, Any],
                   model_name: str = None) -> str:
        """
        Save model with versioning and metadata
        
        MLOps concept: Model artifact management
        - Version control for models
        - Metadata tracking
        - Reproducible model loading
        """
        # Generate model name if not provided
        if model_name is None:
            model_name = f"{model_info['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model directory
        model_path = self.experiment_dir / f"{model_name}.pkl"
        metadata_path = self.experiment_dir / f"{model_name}_metadata.json"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = self.experiment_dir / f"{model_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "experiment_name": self.experiment_name,
            "model_info": model_info,
            "evaluation_results": evaluation_results,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "metadata_path": str(metadata_path)
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Register model
        model_id = str(uuid.uuid4())
        self.model_registry[model_id] = {
            "model_name": model_name,
            "experiment_name": self.experiment_name,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "accuracy": evaluation_results["accuracy"],
            "created_at": datetime.now().isoformat()
        }
        self._save_model_registry()
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Model registered with ID: {model_id}")
        
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """Load a trained model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model ID not found: {model_id}")
        
        model_info = self.model_registry[model_id]
        
        # Load model
        with open(model_info["model_path"], 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = model_info["model_path"].replace('.pkl', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open(model_info["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models"""
        return list(self.model_registry.values())
    
    def run_training_experiment(self,
                               dataset_path: str,
                               model_type: str = "random_forest",
                               hyperparameters: Dict[str, Any] = None,
                               test_size: float = 0.2) -> str:
        """
        Run complete training experiment
        
        This is the main training pipeline that demonstrates MLOps concepts:
        - Reproducible data preparation
        - Model training with hyperparameters
        - Comprehensive evaluation
        - Model versioning and artifact management
        """
        logger.info(f"Starting training experiment: {self.experiment_name}")
        
        # Step 1: Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(
            dataset_path, test_size=test_size
        )
        
        # Step 2: Train model
        model, model_info = self.train_model(
            X_train, y_train, model_type, hyperparameters
        )
        
        # Step 3: Evaluate model
        evaluation_results = self.evaluate_model(model, X_test, y_test, model_info)
        
        # Step 4: Save model
        model_id = self.save_model(model, model_info, evaluation_results)
        
        logger.info(f"Training experiment completed. Model ID: {model_id}")
        
        return model_id


def test_training_pipeline():
    """Test the training pipeline"""
    print("🎯 TESTING TRAINING PIPELINE")
    print("=" * 50)
    
    # Create sample dataset
    print("Creating sample dataset...")
    np.random.seed(42)
    
    # Generate synthetic audio features
    n_samples = 1000
    data = pd.DataFrame({
        'pitch_mean': np.random.normal(200, 50, n_samples),
        'volume_mean': np.random.uniform(0.1, 0.9, n_samples),
        'speech_rate': np.random.uniform(0.3, 0.9, n_samples),
        'silence_ratio': np.random.uniform(0.1, 0.7, n_samples),
        'spectral_centroid_mean': np.random.normal(2000, 500, n_samples),
        'emotion': np.random.choice(['calm', 'anxious', 'frustrated', 'energetic', 'burned_out'], n_samples)
    })
    
    # Save dataset
    dataset_path = "data/emotion_dataset.csv"
    os.makedirs("data", exist_ok=True)
    data.to_csv(dataset_path, index=False)
    
    # Initialize trainer
    trainer = ModelTrainer(experiment_name="emotion_classification_test")
    
    # Run training experiment
    print("\nRunning training experiment...")
    model_id = trainer.run_training_experiment(
        dataset_path=dataset_path,
        model_type="random_forest",
        hyperparameters={"n_estimators": 50, "max_depth": 5}
    )
    
    # Load and test model
    print("\nLoading trained model...")
    model, scaler, metadata = trainer.load_model(model_id)
    
    # Test prediction
    test_features = np.array([[220, 0.6, 0.7, 0.3, 2100]])  # Sample features
    test_features_scaled = scaler.transform(test_features)
    prediction = model.predict(test_features_scaled)
    probability = model.predict_proba(test_features_scaled)
    
    print(f"Test prediction: {prediction[0]}")
    print(f"Prediction probability: {probability[0]}")
    
    # List all models
    print("\nAll trained models:")
    for model_info in trainer.list_models():
        print(f"- {model_info['model_name']}: {model_info['accuracy']:.3f} accuracy")
    
    print("✅ Training pipeline test completed!")


if __name__ == "__main__":
    test_training_pipeline() 