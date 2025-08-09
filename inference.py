"""
Inference Pipeline for MLOps Learning
Learn: model serving, prediction logging, SHAP explanations, monitoring, error handling
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pickle
import librosa
from sklearn.preprocessing import StandardScaler

from utils.audio_features import AudioFeatureExtractor
from utils.sentiment_logger import SentimentLogger
from utils.grafana_observability import GrafanaObservability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Inference pipeline for MLOps learning
    
    Concepts covered:
    - Model serving and loading
    - Prediction logging and monitoring
    - SHAP explanations for interpretability
    - Error handling and fallbacks
    - Performance monitoring
    - Confidence thresholds
    """
    
    def __init__(self,
                 model_id: str,
                 confidence_threshold: float = 0.5,
                 enable_logging: bool = True,
                 enable_grafana: bool = True,
                 enable_shap: bool = True):
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self.enable_grafana = enable_grafana
        self.enable_shap = enable_shap
        
        # Load model and components
        self.model, self.scaler, self.metadata = self._load_model()
        
        # Initialize components
        self.feature_extractor = AudioFeatureExtractor()
        
        if self.enable_logging:
            self.sentiment_logger = SentimentLogger()
        
        if self.enable_grafana:
            self.grafana = GrafanaObservability()
        
        # SHAP explainer (if enabled)
        self.shap_explainer = None
        if self.enable_shap:
            self._initialize_shap()
        
        logger.info(f"InferencePipeline initialized with model: {self.metadata['model_name']}")
    
    def _load_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load trained model and metadata"""
        try:
            # Load model registry
            model_registry_file = Path("models/model_registry.json")
            if not model_registry_file.exists():
                raise FileNotFoundError("Model registry not found")
            
            with open(model_registry_file, 'r') as f:
                model_registry = json.load(f)
            
            if self.model_id not in model_registry:
                raise ValueError(f"Model ID not found: {self.model_id}")
            
            model_info = model_registry[self.model_id]
            
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
            
            logger.info(f"Model loaded: {metadata['model_name']}")
            return model, scaler, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _initialize_shap(self):
        """Initialize SHAP explainer for model interpretability"""
        try:
            import shap
            
            # Create background data for SHAP
            # In practice, you'd use training data samples
            background_data = np.random.randn(100, 5)  # 5 features
            background_data_scaled = self.scaler.transform(background_data)
            
            # Create explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'feature_importances_') else shap.LinearExplainer(self.model, background_data_scaled)
            else:
                logger.warning("Model doesn't support SHAP explanations")
                self.shap_explainer = None
                
        except ImportError:
            logger.warning("SHAP not available, explanations disabled")
            self.shap_explainer = None
        except Exception as e:
            logger.error(f"Error initializing SHAP: {e}")
            self.shap_explainer = None
    
    def extract_audio_features(self, audio_path: str) -> Optional[Dict[str, float]]:
        """
        Extract audio features from file
        
        MLOps concept: Feature extraction pipeline
        - Consistent feature extraction
        - Error handling for corrupted files
        - Performance monitoring
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            
            if not features:
                logger.error("Feature extraction failed")
                return None
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None
    
    def predict_emotion(self, 
                       audio_path: str,
                       speaker_id: str = "unknown",
                       team_id: str = "default") -> Dict[str, Any]:
        """
        Predict emotion from audio file
        
        MLOps concept: Complete inference pipeline
        - Feature extraction
        - Model prediction
        - Confidence scoring
        - SHAP explanations
        - Logging and monitoring
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract features
            feature_extraction_start = time.time()
            features = self.extract_audio_features(audio_path)
            feature_extraction_time = time.time() - feature_extraction_start
            
            if not features:
                return self._create_error_result("Feature extraction failed")
            
            # Step 2: Prepare features for model
            feature_names = self.metadata["model_info"]["feature_names"]
            feature_vector = []
            
            for feature_name in feature_names:
                # Map feature names to actual features
                if "pitch" in feature_name:
                    feature_vector.append(features.get("pitch_mean", 200.0))
                elif "volume" in feature_name:
                    feature_vector.append(features.get("volume_mean", 0.5))
                elif "speech" in feature_name:
                    feature_vector.append(features.get("speech_rate", 0.7))
                elif "silence" in feature_name:
                    feature_vector.append(features.get("silence_ratio", 0.3))
                elif "spectral" in feature_name:
                    feature_vector.append(features.get("spectral_centroid_mean", 2000.0))
                else:
                    feature_vector.append(0.0)  # Default value
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Step 3: Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Step 4: Make prediction
            prediction_start = time.time()
            prediction = self.model.predict(feature_vector_scaled)[0]
            prediction_time = time.time() - prediction_start
            
            # Step 5: Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector_scaled)[0]
                confidence = max(probabilities)
            else:
                probabilities = None
                confidence = 0.5  # Default confidence
            
            # Step 6: Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.warning(f"Low confidence prediction: {confidence:.3f} < {self.confidence_threshold}")
            
            # Step 7: Generate SHAP explanation
            shap_explanation = None
            if self.shap_explainer and self.enable_shap:
                try:
                    shap_values = self.shap_explainer.shap_values(feature_vector_scaled)
                    feature_names = self.metadata["model_info"]["feature_names"]
                    shap_explanation = dict(zip(feature_names, shap_values[0]))
                except Exception as e:
                    logger.error(f"Error generating SHAP explanation: {e}")
            
            # Step 8: Calculate total time
            total_time = time.time() - start_time
            
            # Step 9: Log prediction
            if self.enable_logging:
                self._log_prediction(
                    speaker_id=speaker_id,
                    emotion=prediction,
                    confidence=confidence,
                    features=features,
                    inference_time=total_time,
                    shap_explanation=shap_explanation,
                    team_id=team_id
                )
            
            # Step 10: Send to Grafana
            if self.enable_grafana:
                self._send_to_grafana(
                    emotion=prediction,
                    confidence=confidence,
                    features=features,
                    inference_time=total_time
                )
            
            # Compile results
            result = {
                "success": True,
                "emotion": prediction,
                "confidence": confidence,
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "features": features,
                "shap_explanation": shap_explanation,
                "timing": {
                    "feature_extraction": feature_extraction_time,
                    "prediction": prediction_time,
                    "total": total_time
                },
                "metadata": {
                    "model_id": self.model_id,
                    "model_name": self.metadata["model_name"],
                    "confidence_threshold": self.confidence_threshold,
                    "low_confidence": confidence < self.confidence_threshold
                }
            }
            
            logger.info(f"Prediction completed: {prediction} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._create_error_result(str(e))
    
    def _log_prediction(self,
                       speaker_id: str,
                       emotion: str,
                       confidence: float,
                       features: Dict[str, float],
                       inference_time: float,
                       shap_explanation: Optional[Dict[str, float]],
                       team_id: str):
        """Log prediction for monitoring and debugging"""
        try:
            self.sentiment_logger.log_sentiment_inference(
                speaker_id=speaker_id,
                emotion_prediction=emotion,
                confidence_score=confidence,
                audio_features=features,
                inference_time=inference_time,
                model_name=self.metadata["model_name"],
                team_id=team_id,
                shap_explanation=shap_explanation
            )
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def _send_to_grafana(self,
                         emotion: str,
                         confidence: float,
                         features: Dict[str, float],
                         inference_time: float):
        """Send metrics to Grafana for monitoring"""
        try:
            self.grafana.log_prediction(
                model_name=self.metadata["model_name"],
                emotion=emotion,
                confidence=confidence,
                probabilities={emotion: confidence},
                inference_time=inference_time,
                features=features
            )
        except Exception as e:
            logger.error(f"Error sending to Grafana: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "error": error_message,
            "emotion": "unknown",
            "confidence": 0.0,
            "timestamp": time.time()
        }
    
    def batch_predict(self, 
                     audio_files: List[str],
                     speaker_ids: List[str] = None,
                     team_id: str = "default") -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple audio files
        
        MLOps concept: Batch processing
        - Efficient processing of multiple files
        - Error handling for individual files
        - Performance optimization
        """
        if speaker_ids is None:
            speaker_ids = [f"speaker_{i}" for i in range(len(audio_files))]
        
        if len(audio_files) != len(speaker_ids):
            raise ValueError("Number of audio files must match number of speaker IDs")
        
        results = []
        
        for audio_file, speaker_id in zip(audio_files, speaker_ids):
            try:
                result = self.predict_emotion(audio_file, speaker_id, team_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                results.append(self._create_error_result(str(e)))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "model_name": self.metadata["model_name"],
            "model_type": self.metadata["model_info"]["model_type"],
            "accuracy": self.metadata["evaluation_results"]["accuracy"],
            "feature_names": self.metadata["model_info"]["feature_names"],
            "confidence_threshold": self.confidence_threshold,
            "created_at": self.metadata["created_at"]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check for the inference pipeline
        
        MLOps concept: System health monitoring
        - Model availability
        - Component status
        - Performance metrics
        """
        health_status = {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "feature_extractor_available": self.feature_extractor is not None,
            "logging_enabled": self.enable_logging,
            "grafana_enabled": self.enable_grafana,
            "shap_enabled": self.shap_explainer is not None,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": time.time()
        }
        
        # Test prediction with dummy data
        try:
            dummy_features = np.random.randn(1, len(self.metadata["model_info"]["feature_names"]))
            dummy_features_scaled = self.scaler.transform(dummy_features)
            _ = self.model.predict(dummy_features_scaled)
            health_status["prediction_test"] = True
        except Exception as e:
            health_status["prediction_test"] = False
            health_status["prediction_error"] = str(e)
        
        return health_status


def test_inference_pipeline():
    """Test the inference pipeline"""
    print("🎯 TESTING INFERENCE PIPELINE")
    print("=" * 50)
    
    # First, we need a trained model
    print("Note: This test requires a trained model. Run train.py first to create a model.")
    print("For now, we'll test with a dummy model...")
    
    # Create a dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    # Create dummy model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    
    # Create dummy training data
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice(['calm', 'anxious', 'frustrated', 'energetic', 'burned_out'], 100)
    
    # Fit scaler and model
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    # Save dummy model
    os.makedirs("models/test_experiment", exist_ok=True)
    model_path = "models/test_experiment/dummy_model.pkl"
    scaler_path = "models/test_experiment/dummy_model_scaler.pkl"
    metadata_path = "models/test_experiment/dummy_model_metadata.json"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create metadata
    metadata = {
        "model_name": "dummy_model",
        "model_info": {
            "model_type": "random_forest",
            "feature_names": ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        },
        "evaluation_results": {
            "accuracy": 0.75
        },
        "created_at": datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create model registry entry
    model_registry = {
        "test_model_id": {
            "model_name": "dummy_model",
            "model_path": model_path,
            "metadata_path": metadata_path,
            "accuracy": 0.75,
            "created_at": datetime.now().isoformat()
        }
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/model_registry.json", 'w') as f:
        json.dump(model_registry, f, indent=2)
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(
        model_id="test_model_id",
        confidence_threshold=0.5,
        enable_logging=False,  # Disable for testing
        enable_grafana=False,  # Disable for testing
        enable_shap=True
    )
    
    # Create test audio
    print("Creating test audio...")
    duration = 3
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    test_audio_path = "test_audio.wav"
    import soundfile as sf
    sf.write(test_audio_path, test_audio, sr)
    
    try:
        # Test prediction
        print("\nTesting prediction...")
        result = pipeline.predict_emotion(
            audio_path=test_audio_path,
            speaker_id="test_speaker",
            team_id="test_team"
        )
        
        # Helper function to convert numpy arrays and other types to JSON serializable format
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (bool, int, float, str)) or obj is None:
                return obj
            else:
                return str(obj)  # Convert unknown types to string
        
        result_serializable = convert_numpy(result)
        print(f"Prediction result: {json.dumps(result_serializable, indent=2)}")
        
        # Test health check
        print("\nTesting health check...")
        health = pipeline.health_check()
        health_serializable = convert_numpy(health)
        print(f"Health status: {json.dumps(health_serializable, indent=2)}")
        
        # Test model info
        print("\nModel information:")
        model_info = pipeline.get_model_info()
        model_info_serializable = convert_numpy(model_info)
        print(f"Model info: {json.dumps(model_info_serializable, indent=2)}")
        
        print("✅ Inference pipeline test completed!")
        
    finally:
        # Clean up
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)


if __name__ == "__main__":
    test_inference_pipeline() 