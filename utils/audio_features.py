"""
Audio Feature Extractor
Extracts audio characteristics for emotion detection (pitch, volume, speaking rate, tone)

FEATURE EXTRACTION REASONING
============================

This module extracts audio features designed to support emotion detection and speaking 
feedback analysis. Each feature category serves specific purposes:

1. PITCH FEATURES
   - Purpose: Detect emotional states through voice pitch variations
   - Reasoning: 
     * High pitch (200+ Hz) often indicates excitement, fear, or anger
     * Low pitch (<150 Hz) can indicate sadness or calmness  
     * Pitch variability (std) indicates emotional expressiveness
     * Pitch range shows vocal flexibility and engagement
   - Features: mean, std, range, min, max

2. VOLUME FEATURES  
   - Purpose: Analyze speaking volume and energy levels
   - Reasoning:
     * Loud volume (>0.6) can indicate confidence, anger, or enthusiasm
     * Quiet volume (<0.3) might indicate nervousness or sadness
     * Volume consistency (std) indicates speaking confidence
     * Volume percentiles help identify speaking patterns
   - Features: mean, std, range, min, max, 25th/75th percentiles

3. SPECTRAL FEATURES
   - Purpose: Analyze tone quality and voice characteristics
   - Reasoning:
     * Spectral centroid: Measures "brightness" of voice (higher = more energetic)
     * Spectral rolloff: Indicates frequency distribution and voice quality
     * Spectral bandwidth: Measures frequency spread and voice richness
     * Zero crossing rate: Indicates noisiness/roughness of voice
   - Features: centroid, rolloff, bandwidth, zero crossing rate (mean/std)

4. SPEECH FEATURES
   - Purpose: Analyze speaking patterns, rate, and efficiency
   - Reasoning:
     * Speech rate: Fast speech (>0.8) can indicate nervousness or excitement
     * Silence ratio: Too much silence (>0.7) indicates poor speaking flow
     * Energy variations: Indicates speaking rhythm and confidence
     * Duration ratios: Helps assess speaking efficiency and engagement
   - Features: speech_rate, silence_ratio, energy_variations, duration ratios

INTEGRATION WITH PIPELINE
========================
These features feed into:
- Emotion detection models for classifying emotional states
- Speaking quality assessment for confidence and clarity analysis  
- Real-time feedback generation for live speaking guidance
- Feature summaries for human-readable interpretations

The extractor includes robust error handling and dummy feature fallbacks to ensure
pipeline continuity even without audio processing capabilities.
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    warnings.warn("Librosa not available. Install librosa for audio feature extraction.")


class AudioFeatureExtractor:
    """Extract audio features for emotion detection"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms frames
        self.hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        if not AUDIO_AVAILABLE:
            print("[AudioFeatureExtractor] Warning: Audio libraries not available")
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive audio features
        
        Args:
            audio (np.ndarray): Audio samples
            
        Returns:
            Dict: Extracted audio features
        """
        if not AUDIO_AVAILABLE:
            print("[AudioFeatureExtractor] Error: Audio libraries not available")
            return self._create_dummy_features()
        
        try:
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Extract pitch features
            pitch_features = self._extract_pitch_features(audio)
            
            # Extract volume features
            volume_features = self._extract_volume_features(audio)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(audio)
            
            # Extract speaking rate features
            speech_features = self._extract_speech_features(audio)
            
            # Combine all features
            features = {
                **pitch_features,
                **volume_features,
                **spectral_features,
                **speech_features
            }
            
            print(f"[AudioFeatureExtractor] Extracted {len(features)} features")
            return features
            
        except Exception as e:
            print(f"[AudioFeatureExtractor] Error extracting features: {e}")
            return self._create_dummy_features()
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract pitch-related features for emotion detection
        
        Pitch is a key indicator of emotional state:
        - High pitch (200+ Hz): excitement, fear, anger, enthusiasm
        - Low pitch (<150 Hz): sadness, calmness, depression
        - High variability (std): emotional expressiveness, engagement
        - Wide range: vocal flexibility, dynamic speaking
        - Narrow range: monotone, lack of engagement
        
        Returns:
            Dict containing pitch statistics for emotion analysis
        """
        try:
            # Extract pitch using librosa
            # pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitches, voiced_flag, voiced_probs = librosa.pyin(y=audio, sr=self.sample_rate, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            print(f"[DEBUG] Pitches shape: {pitches.shape}")
            print(f"[DEBUG] Magnitudes shape: {voiced_flag.shape}")
            # Get pitch values where magnitude is above threshold
            pitch_values = [p for i, p in enumerate(pitches) if voiced_flag[i] != 0 and p is not None]
            
            if not pitch_values:
                return {
                    "pitch_mean": 0.0,
                    "pitch_std": 0.0,
                    "pitch_range": 0.0,
                    "pitch_min": 0.0,
                    "pitch_max": 0.0
                }
            
            pitch_values = np.array(pitch_values)
            print(f"[DEBUG] Extracted pitch values: {pitch_values[:10]}... total={len(pitch_values)}")
            
            return {
                "pitch_mean": float(np.mean(pitch_values)),
                "pitch_std": float(np.std(pitch_values)),
                "pitch_range": float(np.max(pitch_values) - np.min(pitch_values)),
                "pitch_min": float(np.min(pitch_values)),
                "pitch_max": float(np.max(pitch_values))
            }
            
        except Exception as e:
            print(f"[AudioFeatureExtractor] Error extracting pitch features: {e}")
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_range": 0.0,
                "pitch_min": 0.0,
                "pitch_max": 0.0
            }
    
    def _extract_volume_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract volume-related features for speaking confidence analysis
        
        Volume characteristics indicate speaking confidence and emotional intensity:
        - High volume (>0.6): confidence, anger, enthusiasm, assertiveness
        - Low volume (<0.3): nervousness, sadness, lack of confidence
        - High variability (std): dynamic speaking, emotional expression
        - Low variability: monotone, lack of engagement
        - Volume percentiles: identify speaking patterns and consistency
        
        Returns:
            Dict containing volume statistics for confidence assessment
        """
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)
            rms = rms.flatten()
            
            # Calculate volume statistics
            volume_mean = float(np.mean(rms))
            volume_std = float(np.std(rms))
            volume_range = float(np.max(rms) - np.min(rms))
            
            # Calculate volume percentiles
            volume_25 = float(np.percentile(rms, 25))
            volume_75 = float(np.percentile(rms, 75))
            
            return {
                "volume_mean": volume_mean,
                "volume_std": volume_std,
                "volume_range": volume_range,
                "volume_min": float(np.min(rms)),
                "volume_max": float(np.max(rms)),
                "volume_25": volume_25,
                "volume_75": volume_75
            }
            
        except Exception as e:
            print(f"[AudioFeatureExtractor] Error extracting volume features: {e}")
            return {
                "volume_mean": 0.0,
                "volume_std": 0.0,
                "volume_range": 0.0,
                "volume_min": 0.0,
                "volume_max": 0.0,
                "volume_25": 0.0,
                "volume_75": 0.0
            }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral features for tone quality and voice characteristics analysis
        
        Spectral features provide insights into voice quality and emotional tone:
        - Spectral centroid: Measures "brightness" (higher = more energetic, enthusiastic)
        - Spectral rolloff: Indicates frequency distribution and voice quality
        - Spectral bandwidth: Measures frequency spread (wider = richer voice)
        - Zero crossing rate: Indicates noisiness/roughness (higher = more emotional/strained)
        
        These features help distinguish between:
        - Clear vs. muffled speech
        - Energetic vs. tired voice
        - Emotional vs. neutral tone
        - Professional vs. casual speaking style
        
        Returns:
            Dict containing spectral statistics for tone analysis
        """
        try:
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ).flatten()
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ).flatten()
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate, hop_length=self.hop_length
            ).flatten()
            
            # Zero crossing rate (noisiness)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio, frame_length=self.frame_length, hop_length=self.hop_length
            ).flatten()
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_centroid_std": float(np.std(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_rolloff_std": float(np.std(spectral_rolloff)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "zero_crossing_rate_std": float(np.std(zero_crossing_rate))
            }
            
        except Exception as e:
            print(f"[AudioFeatureExtractor] Error extracting spectral features: {e}")
            return {
                "spectral_centroid_mean": 0.0,
                "spectral_centroid_std": 0.0,
                "spectral_rolloff_mean": 0.0,
                "spectral_rolloff_std": 0.0,
                "spectral_bandwidth_mean": 0.0,
                "spectral_bandwidth_std": 0.0,
                "zero_crossing_rate_mean": 0.0,
                "zero_crossing_rate_std": 0.0
            }
    
    def _extract_speech_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract speech-related features for speaking pattern and efficiency analysis
        
        Speech features analyze speaking rhythm, rate, and engagement:
        - Speech rate: Fast speech (>0.8) can indicate nervousness or excitement
        - Silence ratio: Too much silence (>0.7) indicates poor speaking flow
        - Energy variations: Indicates speaking rhythm and confidence
        - Duration ratios: Helps assess speaking efficiency and engagement
        
        These features help identify:
        - Speaking confidence and fluency
        - Nervousness vs. enthusiasm
        - Speaking efficiency and engagement
        - Areas for improvement in presentation skills
        
        Returns:
            Dict containing speech pattern statistics for speaking quality assessment
        """
        try:
            # Calculate energy
            energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)
            energy = energy.flatten()
            
            # Detect speech segments (energy above threshold)
            energy_threshold = np.mean(energy) * 0.5
            speech_segments = energy > energy_threshold
            
            # Calculate speech rate
            speech_frames = np.sum(speech_segments)
            total_frames = len(speech_segments)
            speech_rate = speech_frames / total_frames if total_frames > 0 else 0.0
            
            # Calculate silence ratio
            silence_ratio = 1.0 - speech_rate
            
            # Calculate energy variations (speaking rate indicator)
            energy_variations = np.std(np.diff(energy))
            
            # Calculate speech duration
            speech_duration = speech_frames * self.hop_length / self.sample_rate
            total_duration = len(audio) / self.sample_rate
            
            return {
                "speech_rate": float(speech_rate),
                "silence_ratio": float(silence_ratio),
                "energy_variations": float(energy_variations),
                "speech_duration": float(speech_duration),
                "total_duration": float(total_duration),
                "speech_ratio": float(speech_duration / total_duration) if total_duration > 0 else 0.0
            }
            
        except Exception as e:
            print(f"[AudioFeatureExtractor] Error extracting speech features: {e}")
            return {
                "speech_rate": 0.0,
                "silence_ratio": 1.0,
                "energy_variations": 0.0,
                "speech_duration": 0.0,
                "total_duration": 0.0,
                "speech_ratio": 0.0
            }
    
    def _create_dummy_features(self) -> Dict[str, Any]:
        """Create dummy features when audio processing is not available"""
        return {
            # Pitch features
            "pitch_mean": 200.0,
            "pitch_std": 50.0,
            "pitch_range": 100.0,
            "pitch_min": 150.0,
            "pitch_max": 250.0,
            
            # Volume features
            "volume_mean": 0.5,
            "volume_std": 0.2,
            "volume_range": 0.4,
            "volume_min": 0.3,
            "volume_max": 0.7,
            "volume_25": 0.4,
            "volume_75": 0.6,
            
            # Spectral features
            "spectral_centroid_mean": 2000.0,
            "spectral_centroid_std": 500.0,
            "spectral_rolloff_mean": 4000.0,
            "spectral_rolloff_std": 1000.0,
            "spectral_bandwidth_mean": 1500.0,
            "spectral_bandwidth_std": 300.0,
            "zero_crossing_rate_mean": 0.1,
            "zero_crossing_rate_std": 0.05,
            
            # Speech features
            "speech_rate": 0.7,
            "silence_ratio": 0.3,
            "energy_variations": 0.1,
            "speech_duration": 2.1,
            "total_duration": 3.0,
            "speech_ratio": 0.7
        }
    
    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a human-readable summary of extracted features for feedback generation
        
        Converts numerical features into interpretable categories for:
        - Speaking feedback and coaching
        - Emotion detection results
        - Real-time speaking guidance
        - Performance assessment
        
        The summary provides actionable insights for improving speaking skills
        and understanding emotional states during speech.
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            Dict containing categorized feature interpretations
        """
        summary = {
            "pitch_characteristics": {
                "level": "high" if features.get("pitch_mean", 0) > 200 else "low" if features.get("pitch_mean", 0) < 150 else "medium",
                "variability": "high" if features.get("pitch_std", 0) > 50 else "low" if features.get("pitch_std", 0) < 20 else "medium"
            },
            "volume_characteristics": {
                "level": "high" if features.get("volume_mean", 0) > 0.6 else "low" if features.get("volume_mean", 0) < 0.3 else "medium",
                "variability": "high" if features.get("volume_std", 0) > 0.3 else "low" if features.get("volume_std", 0) < 0.1 else "medium"
            },
            "speech_characteristics": {
                "rate": "fast" if features.get("speech_rate", 0) > 0.8 else "slow" if features.get("speech_rate", 0) < 0.4 else "normal",
                "clarity": "clear" if features.get("silence_ratio", 1) < 0.3 else "unclear" if features.get("silence_ratio", 1) > 0.7 else "moderate"
            },
            "tone_characteristics": {
                "brightness": "bright" if features.get("spectral_centroid_mean", 0) > 2500 else "dark" if features.get("spectral_centroid_mean", 0) < 1500 else "medium",
                "noisiness": "noisy" if features.get("zero_crossing_rate_mean", 0) > 0.15 else "clean" if features.get("zero_crossing_rate_mean", 0) < 0.05 else "moderate"
            }
        }
        
        return summary


def test_audio_feature_extractor():
    """Test the audio feature extractor"""
    print("🎵 TESTING AUDIO FEATURE EXTRACTOR")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # Create test audio
    print("\n🎧 Creating test audio...")
    duration = 3  # seconds
    sr = extractor.sample_rate
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Create different types of test audio
    test_audios = {
        "happy": 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t),
        "sad": 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 330 * t),
        "angry": 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t),
        "calm": 0.1 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.sin(2 * np.pi * 400 * t)
    }
    
    for emotion, audio in test_audios.items():
        print(f"\n--- Testing {emotion} audio ---")
        
        # Extract features
        features = extractor.extract_features(audio)
        
        # Get feature summary
        summary = extractor.get_feature_summary(features)
        
        print(f"Features extracted: {len(features)}")
        print(f"Pitch: {summary['pitch_characteristics']}")
        print(f"Volume: {summary['volume_characteristics']}")
        print(f"Speech: {summary['speech_characteristics']}")
        print(f"Tone: {summary['tone_characteristics']}")
    
    print("\n✅ Audio feature extractor test completed!")


if __name__ == "__main__":
    test_audio_feature_extractor() 