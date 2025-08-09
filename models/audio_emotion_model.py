"""
Audio Emotion Detection Model
Analyzes audio characteristics (tone, pitch, volume, speaking rate) to detect emotions
without relying on speech content
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from models.model_factory import BaseModel, DummyModel


class AudioEmotionModel(BaseModel):
    """Model for detecting emotions from audio characteristics"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._audio_processor = None
        self._feature_extractor = None
        print(f"[AudioEmotionModel] Initialized with config: {config}")
    
    def _load_dependencies(self):
        """Load audio processing dependencies"""
        try:
            from utils.audio_utils import AudioProcessor
            from utils.audio_features import AudioFeatureExtractor
            
            self._audio_processor = AudioProcessor(
                sample_rate=self.model_config.get("sample_rate", 16000),
                max_duration=self.model_config.get("max_duration", 30.0)
            )
            
            self._feature_extractor = AudioFeatureExtractor(
                sample_rate=self.model_config.get("sample_rate", 16000)
            )
            
            print("[AudioEmotionModel] Dependencies loaded successfully")
            return True
            
        except ImportError as e:
            print(f"[AudioEmotionModel] Warning: Dependencies not available: {e}")
            return False
    
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict emotion from audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            Dict: Emotion analysis result
        """
        print(f"[AudioEmotionModel] Received audio: '{audio_path}'")
        
        # Load dependencies if not already loaded
        if self._audio_processor is None:
            if not self._load_dependencies():
                return self._create_error_result(audio_path, "Dependencies not available")
        
        try:
            # Step 1: Load and preprocess audio
            print(f"[AudioEmotionModel] Loading and preprocessing audio...")
            audio = self._audio_processor.load_audio(audio_path)
            
            if audio is None:
                return self._create_error_result(audio_path, "Audio loading failed")
            
            # Step 2: Extract audio features
            print(f"[AudioEmotionModel] Extracting audio features...")
            features = self._feature_extractor.extract_features(audio)
            
            # Step 3: Analyze audio characteristics
            print(f"[AudioEmotionModel] Analyzing audio characteristics...")
            analysis = self._analyze_audio_characteristics(audio, features)
            
            # Step 4: Predict emotion based on characteristics
            print(f"[AudioEmotionModel] Predicting emotion...")
            emotion_result = self._predict_emotion_from_features(features, analysis)
            
            # Add audio-specific metadata
            result = {
                **emotion_result,
                "input_type": "audio",
                "audio_path": audio_path,
                "audio_features": features,
                "audio_analysis": analysis,
                "audio_samples": len(audio),
                "audio_duration": len(audio) / self._audio_processor.sample_rate,
                "processing_pipeline": ["audio_load", "feature_extraction", "emotion_analysis"]
            }
            
            print(f"[AudioEmotionModel] Returning: {result}")
            return result
            
        except Exception as e:
            print(f"[AudioEmotionModel] Error during prediction: {e}")
            return self._create_error_result(audio_path, str(e))
    
    def batch_predict(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict emotion for multiple audio files
        
        Args:
            audio_paths (List[str]): List of audio file paths
            
        Returns:
            List[Dict]: List of emotion analysis results
        """
        print(f"[AudioEmotionModel] Received batch of {len(audio_paths)} audio files")
        
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"[AudioEmotionModel] Error processing {audio_path}: {e}")
                results.append(self._create_error_result(audio_path, str(e)))
        
        print(f"[AudioEmotionModel] Returning batch results: {results}")
        return results
    
    def _analyze_audio_characteristics(self, audio: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze audio characteristics for emotion detection
        
        Args:
            audio (np.ndarray): Audio samples
            features (Dict): Extracted audio features
            
        Returns:
            Dict: Audio characteristics analysis
        """
        # Analyze speaking rate (based on energy variations)
        energy = np.mean(audio**2)
        energy_variations = np.std(np.diff(audio**2))
        
        # Analyze pitch characteristics
        pitch_mean = features.get("pitch_mean", 0)
        pitch_std = features.get("pitch_std", 0)
        pitch_range = features.get("pitch_range", 0)
        
        # Analyze volume characteristics
        volume_mean = features.get("volume_mean", 0)
        volume_std = features.get("volume_std", 0)
        volume_range = features.get("volume_range", 0)
        
        # Analyze tone characteristics
        spectral_centroid = features.get("spectral_centroid", 0)
        spectral_rolloff = features.get("spectral_rolloff", 0)
        
        # Analyze speaking patterns
        silence_ratio = features.get("silence_ratio", 0)
        speech_rate = features.get("speech_rate", 0)
        
        analysis = {
            "speaking_rate": {
                "energy": energy,
                "energy_variations": energy_variations,
                "speech_rate": speech_rate,
                "silence_ratio": silence_ratio
            },
            "pitch_characteristics": {
                "mean": pitch_mean,
                "std": pitch_std,
                "range": pitch_range,
                "variability": pitch_std / (pitch_mean + 1e-6)
            },
            "volume_characteristics": {
                "mean": volume_mean,
                "std": volume_std,
                "range": volume_range,
                "variability": volume_std / (volume_mean + 1e-6)
            },
            "tone_characteristics": {
                "spectral_centroid": spectral_centroid,
                "spectral_rolloff": spectral_rolloff,
                "brightness": spectral_centroid / 8000.0  # Normalized brightness
            }
        }
        
        print(f"[AudioEmotionModel] Audio analysis completed")
        return analysis
    
    def _predict_emotion_from_features(self, features: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict emotion based on audio features and characteristics
        
        Args:
            features (Dict): Extracted audio features
            analysis (Dict): Audio characteristics analysis
            
        Returns:
            Dict: Emotion prediction result
        """
        # Extract key characteristics
        pitch_mean = analysis["pitch_characteristics"]["mean"]
        pitch_variability = analysis["pitch_characteristics"]["variability"]
        volume_mean = analysis["volume_characteristics"]["mean"]
        volume_variability = analysis["volume_characteristics"]["variability"]
        brightness = analysis["tone_characteristics"]["brightness"]
        speech_rate = analysis["speaking_rate"]["speech_rate"]
        energy_variations = analysis["speaking_rate"]["energy_variations"]
        
        # Emotion prediction logic based on audio characteristics
        emotions = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "calm": 0.0,
            "excited": 0.0,
            "anxious": 0.0,
            "neutral": 0.0
        }
        
        # High pitch + high variability + high volume = Happy/Excited
        if pitch_mean > 200 and pitch_variability > 0.3 and volume_mean > 0.5:
            emotions["happy"] += 0.4
            emotions["excited"] += 0.3
        
        # Low pitch + low variability + low volume = Sad/Calm
        if pitch_mean < 150 and pitch_variability < 0.2 and volume_mean < 0.3:
            emotions["sad"] += 0.4
            emotions["calm"] += 0.3
        
        # High volume + high variability + fast speech = Angry
        if volume_mean > 0.7 and volume_variability > 0.4 and speech_rate > 0.8:
            emotions["angry"] += 0.5
        
        # High pitch variability + high energy variations = Anxious
        if pitch_variability > 0.4 and energy_variations > 0.1:
            emotions["anxious"] += 0.4
        
        # Balanced characteristics = Neutral
        if (0.1 < pitch_variability < 0.3 and 
            0.2 < volume_variability < 0.4 and 
            0.3 < brightness < 0.7):
            emotions["neutral"] += 0.4
        
        # Normalize scores
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        
        # Get primary emotion
        primary_emotion = max(emotions, key=emotions.get)
        confidence = emotions[primary_emotion]
        
        # Ensure minimum confidence
        if confidence < 0.2:
            primary_emotion = "neutral"
            confidence = 0.5
        
        result = {
            "emotion": primary_emotion,
            "confidence": confidence,
            "emotion_scores": emotions,
            "audio_characteristics": {
                "pitch_level": "high" if pitch_mean > 200 else "low" if pitch_mean < 150 else "medium",
                "volume_level": "high" if volume_mean > 0.6 else "low" if volume_mean < 0.3 else "medium",
                "speaking_rate": "fast" if speech_rate > 0.8 else "slow" if speech_rate < 0.4 else "normal",
                "tone_brightness": "bright" if brightness > 0.6 else "dark" if brightness < 0.3 else "medium"
            }
        }
        
        print(f"[AudioEmotionModel] Predicted emotion: {primary_emotion} (confidence: {confidence:.3f})")
        return result
    
    def _create_error_result(self, audio_path: str, error_message: str) -> Dict[str, Any]:
        """Create error result when processing fails"""
        return {
            "emotion": "error",
            "confidence": 0.0,
            "text": f"Error processing audio: {error_message}",
            "input_type": "audio",
            "audio_path": audio_path,
            "error": error_message,
            "processing_pipeline": ["error"]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "audio_emotion",
            "version": self.version,
            "confidence_threshold": self.confidence_threshold,
            "emotion_labels": ["happy", "sad", "angry", "calm", "excited", "anxious", "neutral"],
            "supports_audio": True,
            "content_independent": True,
            "analyzes_characteristics": ["pitch", "volume", "speaking_rate", "tone", "energy"]
        }


def test_audio_emotion_model():
    """Test the audio emotion model"""
    print("🎵 TESTING AUDIO EMOTION MODEL")
    print("=" * 60)
    
    # Create test configuration
    config = {
        "type": "audio_emotion",
        "version": "v1.0",
        "confidence_threshold": 0.5,
        "emotion_labels": ["happy", "sad", "angry", "calm", "excited", "anxious", "neutral"],
        "config": {
            "sample_rate": 16000,
            "max_duration": 30.0,
            "supported_formats": ["wav", "mp3", "flac"]
        }
    }
    
    # Initialize model
    model = AudioEmotionModel(config)
    
    # Test with dummy audio
    from utils.audio_utils import AudioProcessor
    audio_processor = AudioProcessor()
    
    print("\n🎧 Creating test audio...")
    test_audio_path = "test_emotion_audio.wav"
    
    try:
        if audio_processor.AUDIO_AVAILABLE:
            # Generate different types of test audio
            duration = 3  # seconds
            sr = audio_processor.sample_rate
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            
            # Create "happy" audio (high pitch, bright tone)
            happy_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
            audio_processor.save_audio(happy_audio, test_audio_path, sr)
            
            # Test single prediction
            print(f"\n🔍 Testing single audio emotion prediction...")
            result = model.predict(test_audio_path)
            print(f"Result: {result}")
            
            # Test batch prediction
            print(f"\n🔍 Testing batch audio emotion prediction...")
            batch_results = model.batch_predict([test_audio_path, test_audio_path])
            print(f"Batch results: {batch_results}")
            
        else:
            print("❌ Audio libraries not available, skipping audio emotion test")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            print(f"Cleaned up: {test_audio_path}")
    
    print("\n✅ Audio emotion model test completed!")


if __name__ == "__main__":
    test_audio_emotion_model() 