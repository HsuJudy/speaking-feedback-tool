"""
Audio processing utilities for NeMo speech recognition models
Handles audio file loading, preprocessing, and format conversion
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    warnings.warn("Audio processing libraries not available. Install librosa and soundfile for audio support.")


class AudioProcessor:
    """Audio processing utilities for speech recognition"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        self.AUDIO_AVAILABLE = AUDIO_AVAILABLE
        
        if not AUDIO_AVAILABLE:
            print("[AudioProcessor] Warning: Audio libraries not available")
    
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load audio file and convert to required format
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            np.ndarray: Audio samples at target sample rate
        """
        if not AUDIO_AVAILABLE:
            print(f"[AudioProcessor] Cannot load {file_path}: audio libraries not available")
            return None
        
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Trim to max duration
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
                print(f"[AudioProcessor] Trimmed audio to {self.max_duration}s")
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            print(f"[AudioProcessor] Loaded {file_path}: {len(audio)} samples at {sr}Hz")
            return audio
            
        except Exception as e:
            print(f"[AudioProcessor] Error loading {file_path}: {e}")
            return None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for speech recognition
        
        Args:
            audio (np.ndarray): Raw audio samples
            
        Returns:
            np.ndarray: Preprocessed audio
        """
        if not AUDIO_AVAILABLE:
            return audio
        
        try:
            # Apply pre-emphasis filter
            pre_emphasis = 0.97
            emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Apply noise reduction (simple high-pass filter)
            from scipy import signal
            b, a = signal.butter(4, 100, 'hp', fs=self.sample_rate)
            filtered_audio = signal.filtfilt(b, a, emphasized_audio)
            
            print(f"[AudioProcessor] Preprocessed audio: {len(filtered_audio)} samples")
            return filtered_audio
            
        except Exception as e:
            print(f"[AudioProcessor] Error preprocessing audio: {e}")
            return audio
    
    def create_dummy_audio(self, text: str, duration: float = 5.0) -> np.ndarray:
        """
        Create dummy audio for testing (simulates speech)
        
        Args:
            text (str): Text to simulate
            duration (float): Duration in seconds
            
        Returns:
            np.ndarray: Dummy audio samples
        """
        # Create a simple sine wave as dummy audio
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create different frequencies based on text sentiment
        if any(word in text.lower() for word in ["love", "amazing", "perfect", "delicious", "great"]):
            # Higher frequency for positive sentiment
            frequency = 440  # A4 note
        elif any(word in text.lower() for word in ["worst", "terrible", "bad", "hate", "awful"]):
            # Lower frequency for negative sentiment
            frequency = 220  # A3 note
        else:
            # Medium frequency for neutral sentiment
            frequency = 330  # E4 note
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio + noise
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        print(f"[AudioProcessor] Created dummy audio: {len(audio)} samples at {frequency}Hz")
        return audio
    
    def audio_to_text_simulation(self, audio: np.ndarray, original_text: str) -> str:
        """
        Simulate speech-to-text conversion
        
        Args:
            audio (np.ndarray): Audio samples
            original_text (str): Original text (for simulation)
            
        Returns:
            str: Simulated transcribed text
        """
        # In a real scenario, this would use the NeMo model for transcription
        # For now, we simulate by returning the original text with some variations
        
        print(f"[AudioProcessor] Simulating speech-to-text for {len(audio)} samples")
        
        # Simulate transcription errors based on audio quality
        if len(audio) < 1000:  # Very short audio
            return "..."  # Simulate unclear speech
        elif len(audio) < 5000:  # Short audio
            return original_text[:len(original_text)//2] + "..."  # Partial transcription
        else:
            return original_text  # Full transcription
    
    def get_audio_info(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Get information about audio
        
        Args:
            audio (np.ndarray): Audio samples
            
        Returns:
            Dict: Audio information
        """
        duration = len(audio) / self.sample_rate
        rms = np.sqrt(np.mean(audio**2))
        
        return {
            "duration": round(duration, 2),
            "sample_rate": self.sample_rate,
            "samples": len(audio),
            "rms_level": round(rms, 4),
            "max_amplitude": round(np.max(np.abs(audio)), 4)
        }


def test_audio_processor():
    """Test the audio processor"""
    print("🎵 TESTING AUDIO PROCESSOR")
    print("=" * 40)
    
    processor = AudioProcessor()
    
    # Test dummy audio creation
    test_text = "I love this product! It's amazing!"
    dummy_audio = processor.create_dummy_audio(test_text, duration=3.0)
    
    # Get audio info
    info = processor.get_audio_info(dummy_audio)
    print(f"Audio info: {info}")
    
    # Test preprocessing
    processed_audio = processor.preprocess_audio(dummy_audio)
    print(f"Preprocessed audio length: {len(processed_audio)}")
    
    # Test transcription simulation
    transcribed_text = processor.audio_to_text_simulation(processed_audio, test_text)
    print(f"Original text: {test_text}")
    print(f"Transcribed text: {transcribed_text}")
    
    print("✅ Audio processor test completed!")


if __name__ == "__main__":
    test_audio_processor() 