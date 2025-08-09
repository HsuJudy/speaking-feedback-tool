"""
Sentiment Observability Pipeline
Complete pipeline for team sentiment analysis with real-time logging and visualization
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.audio_features import AudioFeatureExtractor
from utils.sentiment_logger import SentimentLogger
from utils.mood_map import DynamicMoodMap
from utils.grafana_observability import GrafanaObservability
from models.audio_emotion_model import AudioEmotionModel
from models.model_factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentObservabilityPipeline:
    """Complete sentiment observability pipeline for team calls and voice clips"""
    
    def __init__(self,
                 model_name: str = "audio_emotion",
                 enable_grafana: bool = True,
                 enable_mood_map: bool = True,
                 db_path: str = "sentiment_logs.db"):
        self.model_name = model_name
        self.enable_grafana = enable_grafana
        self.enable_mood_map = enable_mood_map
        
        # Initialize components
        self._init_components(db_path)
        
        logger.info("SentimentObservabilityPipeline initialized")
    
    def _init_components(self, db_path: str):
        """Initialize all pipeline components"""
        # Audio feature extractor
        self.audio_extractor = AudioFeatureExtractor()
        
        # Sentiment logger
        self.sentiment_logger = SentimentLogger(
            db_path=db_path,
            enable_shap=True,
            enable_burnout_detection=True
        )
        
        # Mood map
        if self.enable_mood_map:
            self.mood_map = DynamicMoodMap(grid_size=(5, 5))
        else:
            self.mood_map = None
        
        # Grafana observability
        if self.enable_grafana:
            self.grafana = GrafanaObservability(enabled=True)
        else:
            self.grafana = None
        
        # Emotion model
        try:
            self.emotion_model = ModelFactory.create_model(self.model_name)
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            self.emotion_model = None
    
    def process_audio_input(self,
                           audio_path: str,
                           speaker_id: str,
                           team_id: str = "default",
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio input through the complete pipeline
        
        Args:
            audio_path: Path to audio file
            speaker_id: Speaker identifier
            team_id: Team identifier
            session_id: Optional session identifier
            
        Returns:
            Dict containing complete analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing audio for speaker {speaker_id} in team {team_id}")
            
            # Step 1: Extract audio features
            logger.info("Step 1: Extracting audio features...")
            audio_features = self._extract_audio_features(audio_path)
            
            if not audio_features:
                return self._create_error_result("Audio feature extraction failed")
            
            # Step 2: Predict emotion
            logger.info("Step 2: Predicting emotion...")
            emotion_result = self._predict_emotion(audio_path, audio_features)
            
            if not emotion_result:
                return self._create_error_result("Emotion prediction failed")
            
            # Step 3: Calculate additional metrics
            logger.info("Step 3: Calculating metrics...")
            burnout_risk = self._calculate_burnout_risk(emotion_result, audio_features)
            sentiment_drift = self._calculate_sentiment_drift(speaker_id, emotion_result)
            
            # Step 4: Log sentiment inference
            logger.info("Step 4: Logging sentiment inference...")
            log_session_id = self._log_sentiment_inference(
                speaker_id=speaker_id,
                emotion_result=emotion_result,
                audio_features=audio_features,
                burnout_risk=burnout_risk,
                team_id=team_id,
                session_id=session_id
            )
            
            # Step 5: Update mood map
            if self.mood_map:
                logger.info("Step 5: Updating mood map...")
                mood_map_data = self._update_mood_map(
                    speaker_id=speaker_id,
                    emotion_result=emotion_result,
                    burnout_risk=burnout_risk,
                    team_id=team_id
                )
            else:
                mood_map_data = None
            
            # Step 6: Send to Grafana
            if self.grafana:
                logger.info("Step 6: Sending to Grafana...")
                self._send_to_grafana(
                    speaker_id=speaker_id,
                    emotion_result=emotion_result,
                    audio_features=audio_features,
                    burnout_risk=burnout_risk,
                    team_id=team_id
                )
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Compile results
            result = {
                "success": True,
                "speaker_id": speaker_id,
                "team_id": team_id,
                "session_id": log_session_id,
                "emotion_prediction": emotion_result.get("emotion", "unknown"),
                "confidence_score": emotion_result.get("confidence", 0.0),
                "burnout_risk_score": burnout_risk,
                "sentiment_drift_score": sentiment_drift,
                "audio_features": audio_features,
                "mood_map_data": mood_map_data,
                "processing_time": total_time,
                "timestamp": time.time()
            }
            
            logger.info(f"Pipeline completed successfully in {total_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._create_error_result(str(e))
    
    def _extract_audio_features(self, audio_path: str) -> Optional[Dict[str, float]]:
        """Extract audio features from file"""
        try:
            # Load audio using librosa
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            features = self.audio_extractor.extract_features(audio)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None
    
    def _predict_emotion(self, audio_path: str, audio_features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Predict emotion from audio"""
        try:
            if self.emotion_model:
                # Use the emotion model
                result = self.emotion_model.predict(audio_path)
                return result
            else:
                # Fallback to feature-based prediction
                return self._predict_emotion_from_features(audio_features)
                
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return None
    
    def _predict_emotion_from_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict emotion based on audio features"""
        # Simple rule-based emotion prediction
        pitch_mean = features.get("pitch_mean", 200)
        volume_mean = features.get("volume_mean", 0.5)
        speech_rate = features.get("speech_rate", 0.7)
        silence_ratio = features.get("silence_ratio", 0.3)
        
        # Determine emotion based on features
        if volume_mean > 0.7 and pitch_mean > 250:
            emotion = "energetic"
            confidence = 0.85
        elif volume_mean < 0.3 and pitch_mean < 150:
            emotion = "burned_out"
            confidence = 0.80
        elif silence_ratio > 0.5 and speech_rate < 0.4:
            emotion = "anxious"
            confidence = 0.75
        elif volume_mean > 0.6 and pitch_mean > 200:
            emotion = "frustrated"
            confidence = 0.70
        else:
            emotion = "calm"
            confidence = 0.65
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "features_used": list(features.keys())
        }
    
    def _calculate_burnout_risk(self, emotion_result: Dict[str, Any], audio_features: Dict[str, float]) -> float:
        """Calculate burnout risk score"""
        emotion = emotion_result.get("emotion", "calm")
        confidence = emotion_result.get("confidence", 0.5)
        
        # Base risk from emotion
        emotion_risk = {
            "burned_out": 0.9,
            "frustrated": 0.7,
            "anxious": 0.6,
            "calm": 0.2,
            "energetic": 0.1
        }.get(emotion, 0.5)
        
        # Audio feature contributions
        audio_risk = 0.0
        
        if audio_features.get("volume_mean", 0) < 0.3:
            audio_risk += 0.2
        
        if audio_features.get("pitch_std", 0) < 20:
            audio_risk += 0.15
        
        if audio_features.get("speech_rate", 0) < 0.4:
            audio_risk += 0.1
        
        if audio_features.get("silence_ratio", 0) > 0.5:
            audio_risk += 0.15
        
        # Combine with confidence
        final_risk = (emotion_risk * 0.6 + audio_risk * 0.4) * confidence
        return min(final_risk, 1.0)
    
    def _calculate_sentiment_drift(self, speaker_id: str, emotion_result: Dict[str, Any]) -> float:
        """Calculate sentiment drift score"""
        # This would typically use historical data
        # For now, return a simple calculation
        emotion = emotion_result.get("emotion", "calm")
        confidence = emotion_result.get("confidence", 0.5)
        
        emotion_scores = {
            "energetic": 1.0,
            "calm": 0.5,
            "anxious": -0.3,
            "frustrated": -0.7,
            "burned_out": -1.0
        }
        
        return emotion_scores.get(emotion, 0.0) * confidence
    
    def _log_sentiment_inference(self,
                                speaker_id: str,
                                emotion_result: Dict[str, Any],
                                audio_features: Dict[str, float],
                                burnout_risk: float,
                                team_id: str,
                                session_id: Optional[str]) -> str:
        """Log sentiment inference"""
        return self.sentiment_logger.log_sentiment_inference(
            speaker_id=speaker_id,
            emotion_prediction=emotion_result.get("emotion", "unknown"),
            confidence_score=emotion_result.get("confidence", 0.0),
            audio_features=audio_features,
            inference_time=emotion_result.get("inference_time", 0.0),
            model_name=self.model_name,
            session_id=session_id,
            team_id=team_id,
            shap_explanation=emotion_result.get("shap_explanation", {})
        )
    
    def _update_mood_map(self,
                        speaker_id: str,
                        emotion_result: Dict[str, Any],
                        burnout_risk: float,
                        team_id: str) -> Optional[Dict[str, Any]]:
        """Update mood map with new data"""
        if not self.mood_map:
            return None
        
        return self.mood_map.update_speaker_mood(
            speaker_id=speaker_id,
            emotion=emotion_result.get("emotion", "unknown"),
            confidence=emotion_result.get("confidence", 0.0),
            burnout_risk=burnout_risk,
            team_id=team_id
        )
    
    def _send_to_grafana(self,
                         speaker_id: str,
                         emotion_result: Dict[str, Any],
                         audio_features: Dict[str, float],
                         burnout_risk: float,
                         team_id: str):
        """Send metrics to Grafana"""
        if not self.grafana:
            return
        
        try:
            self.grafana.log_prediction(
                model_name=self.model_name,
                emotion=emotion_result.get("emotion", "unknown"),
                confidence=emotion_result.get("confidence", 0.0),
                probabilities={emotion_result.get("emotion", "unknown"): emotion_result.get("confidence", 0.0)},
                inference_time=emotion_result.get("inference_time", 0.0),
                features=audio_features
            )
        except Exception as e:
            logger.error(f"Error sending to Grafana: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "success": False,
            "error": error_message,
            "timestamp": time.time()
        }
    
    def get_team_summary(self, team_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get team sentiment summary"""
        return self.sentiment_logger.get_team_sentiment_summary(team_id, hours)
    
    def get_speaker_breakdown(self, team_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get speaker-level breakdown"""
        return self.sentiment_logger.get_speaker_breakdown(team_id, hours)
    
    def get_mood_map_data(self, team_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get current mood map data"""
        if self.mood_map:
            return self.mood_map.get_mood_map_data(team_id)
        return None
    
    def get_team_comparison(self, team_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Compare multiple teams"""
        if self.mood_map:
            return self.mood_map.get_team_comparison(team_ids)
        return None


def test_sentiment_observability_pipeline():
    """Test the complete sentiment observability pipeline"""
    print("🎯 TESTING SENTIMENT OBSERVABILITY PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = SentimentObservabilityPipeline(
        model_name="dummy",  # Use dummy model for testing
        enable_grafana=False,  # Disable for testing
        enable_mood_map=True
    )
    
    # Create test audio file
    print("Creating test audio...")
    import numpy as np
    import librosa
    
    # Generate test audio
    duration = 3
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    # Save test audio
    test_audio_path = "test_audio.wav"
    import soundfile as sf
    sf.write(test_audio_path, test_audio, sr)
    
    try:
        # Test pipeline processing
        print("\nProcessing audio through pipeline...")
        result = pipeline.process_audio_input(
            audio_path=test_audio_path,
            speaker_id="test_speaker_001",
            team_id="test_team_a"
        )
        
        print(f"Pipeline Result: {json.dumps(result, indent=2)}")
        
        # Test team summary
        print("\nGetting team summary...")
        summary = pipeline.get_team_summary("test_team_a", hours=1)
        print(f"Team Summary: {json.dumps(summary, indent=2)}")
        
        # Test mood map
        print("\nGetting mood map data...")
        mood_data = pipeline.get_mood_map_data("test_team_a")
        print(f"Mood Map Data: {json.dumps(mood_data, indent=2)}")
        
        print("✅ Sentiment observability pipeline test completed!")
        
    finally:
        # Clean up test file
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)


if __name__ == "__main__":
    test_sentiment_observability_pipeline() 