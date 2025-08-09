"""
Custom Emotion Recognition Model Trainer
Trains custom models on RAVDESS dataset with observability features
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import time
from datetime import datetime

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Observability imports
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    shap_base_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[List[str]] = None
    confidence_scores: Optional[np.ndarray] = None
    cross_val_scores: Optional[List[float]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class CustomEmotionTrainer:
    """Train custom emotion recognition models"""
    
    def __init__(self, data_dir: str = "data/ravdess", models_dir: str = "models/custom"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Available models
        self.models = {
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, multi_class='multinomial'
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42, learning_rate=0.1
            )
        }
        
        # Feature columns (excluding metadata)
        self.feature_columns = [
            'duration', 'sample_rate', 'rms_energy', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3'
        ]
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Metrics storage
        self.metrics_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data
        
        Returns:
            Tuple: (X, y) features and labels
        """
        features_path = self.data_dir / "features.csv"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # Load data
        df = pd.read_csv(features_path)
        
        # Filter out samples with missing features
        df = df.dropna(subset=self.feature_columns)
        
        # Prepare features and labels
        X = df[self.feature_columns].copy()
        y = df['emotion'].copy()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Loaded {len(X)} samples with {len(self.feature_columns)} features")
        self.logger.info(f"Label distribution: {dict(zip(y.unique(), np.bincount(y_encoded)))}")
        
        return X, pd.Series(y_encoded, index=y.index)
    
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, ModelMetrics]:
        """
        Train a specific model
        
        Args:
            model_name (str): Name of model to train
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            
        Returns:
            Tuple: (trained_model, metrics)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For logistic regression, use absolute coefficients
            feature_importance = dict(zip(self.feature_columns, np.abs(model.coef_[0])))
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        )
        
        # SHAP analysis
        shap_values = None
        shap_base_values = None
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.LinearExplainer(model, X_train_scaled)
                shap_values = explainer.shap_values(X_test_scaled)
                shap_base_values = explainer.expected_value
            except Exception as e:
                self.logger.warning(f"SHAP analysis failed: {e}")
        
        # Create metrics
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            f1_score=report['weighted avg']['f1-score'],
            training_time=training_time,
            inference_time=inference_time,
            feature_importance=feature_importance,
            shap_values=shap_values,
            shap_base_values=shap_base_values,
            shap_feature_names=self.feature_columns,
            confidence_scores=np.max(y_pred_proba, axis=1),
            cross_val_scores=cv_scores.tolist()
        )
        
        self.logger.info(f"Trained {model_name}: accuracy={accuracy:.3f}, f1={metrics.f1_score:.3f}")
        
        return model, metrics
    
    def train_all_models(self) -> Dict[str, Tuple[Any, ModelMetrics]]:
        """
        Train all available models
        
        Returns:
            Dict: Model name -> (model, metrics)
        """
        # Load data
        X, y = self.load_data()
        
        results = {}
        
        for model_name in self.models.keys():
            self.logger.info(f"Training {model_name}...")
            
            try:
                model, metrics = self.train_model(model_name, X, y)
                results[model_name] = (model, metrics)
                self.metrics_history.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
        
        return results
    
    def save_model(self, model: Any, model_name: str, metrics: ModelMetrics) -> str:
        """
        Save trained model to pickle file
        
        Args:
            model: Trained model
            model_name (str): Name of model
            metrics (ModelMetrics): Model metrics
            
        Returns:
            str: Path to saved model
        """
        # Create model package
        model_package = {
            'model': model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'metrics': asdict(metrics),
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to pickle
        model_path = self.models_dir / f"{model_name}_emotion_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        self.logger.info(f"Saved model to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load trained model from pickle file
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            Dict: Model package
        """
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.logger.info(f"Loaded model from {model_path}")
        return model_package
    
    def predict_emotion(self, model_package: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict emotion using loaded model
        
        Args:
            model_package (Dict): Loaded model package
            features (Dict): Audio features
            
        Returns:
            Dict: Prediction results
        """
        model = model_package['model']
        scaler = model_package['scaler']
        label_encoder = model_package['label_encoder']
        
        # Prepare features
        feature_vector = np.array([features.get(col, 0.0) for col in self.feature_columns])
        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
        
        # Predict
        prediction = model.predict(feature_vector_scaled)[0]
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        confidence = np.max(probabilities)
        
        # Decode prediction
        emotion = label_encoder.inverse_transform([prediction])[0]
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, prob in enumerate(probabilities):
            emotion_name = label_encoder.inverse_transform([i])[0]
            emotion_probs[emotion_name] = float(prob)
        
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'probabilities': emotion_probs,
            'model_name': model_package['model_name'],
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_observability_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive observability report
        
        Returns:
            Dict: Observability report
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Aggregate metrics
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.metrics_history),
            "models": {},
            "comparison": {
                "best_accuracy": None,
                "best_f1_score": None,
                "fastest_training": None,
                "fastest_inference": None
            }
        }
        
        best_accuracy = 0
        best_f1 = 0
        fastest_training = float('inf')
        fastest_inference = float('inf')
        
        for metrics in self.metrics_history:
            model_data = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "training_time": metrics.training_time,
                "inference_time": metrics.inference_time,
                "cross_val_mean": np.mean(metrics.cross_val_scores),
                "cross_val_std": np.std(metrics.cross_val_scores),
                "feature_importance": metrics.feature_importance,
                "timestamp": metrics.timestamp
            }
            
            report["models"][metrics.model_name] = model_data
            
            # Track best performers
            if metrics.accuracy > best_accuracy:
                best_accuracy = metrics.accuracy
                report["comparison"]["best_accuracy"] = metrics.model_name
            
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                report["comparison"]["best_f1_score"] = metrics.model_name
            
            if metrics.training_time < fastest_training:
                fastest_training = metrics.training_time
                report["comparison"]["fastest_training"] = metrics.model_name
            
            if metrics.inference_time < fastest_inference:
                fastest_inference = metrics.inference_time
                report["comparison"]["fastest_inference"] = metrics.model_name
        
        return report
    
    def save_observability_report(self, report: Dict[str, Any], filename: str = "observability_report.json"):
        """
        Save observability report to file
        
        Args:
            report (Dict): Observability report
            filename (str): Output filename
        """
        report_path = self.models_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Saved observability report to {report_path}")


def main():
    """Main training function"""
    trainer = CustomEmotionTrainer()
    
    print("🎭 TRAINING CUSTOM EMOTION MODELS")
    print("=" * 50)
    
    # Train all models
    results = trainer.train_all_models()
    
    if not results:
        print("❌ No models were trained successfully")
        return
    
    # Save models
    for model_name, (model, metrics) in results.items():
        model_path = trainer.save_model(model, model_name, metrics)
        print(f"✅ Saved {model_name} to {model_path}")
    
    # Generate observability report
    report = trainer.generate_observability_report()
    trainer.save_observability_report(report)
    
    print("\n📊 MODEL COMPARISON:")
    print("=" * 30)
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


if __name__ == "__main__":
    main() 