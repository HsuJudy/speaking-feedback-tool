"""
Video Sentiment Model
Processes video files by extracting audio and analyzing sentiment
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from models.model_factory import BaseModel, DummyModel


class VideoSentimentModel(BaseModel):
    """Model for processing video files and analyzing sentiment from audio"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._audio_model = None
        self._video_processor = None
        print(f"[VideoSentimentModel] Initialized with config: {config}")
    
    def _load_dependencies(self):
        """Load video and audio processing dependencies"""
        try:
            from utils.video_utils import VideoProcessor
            from utils.audio_utils import AudioProcessor
            
            self._video_processor = VideoProcessor(
                sample_rate=self.model_config.get("sample_rate", 16000),
                max_duration=self.model_config.get("max_duration", 30.0)
            )
            
            self._audio_model = DummyModel(self.config)  # Use dummy for now
            print("[VideoSentimentModel] Dependencies loaded successfully")
            
        except ImportError as e:
            print(f"[VideoSentimentModel] Warning: Dependencies not available: {e}")
            return False
        
        return True
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict sentiment from video file
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Dict: Sentiment analysis result
        """
        print(f"[VideoSentimentModel] Received video: '{video_path}'")
        
        # Load dependencies if not already loaded
        if self._video_processor is None:
            if not self._load_dependencies():
                return self._create_error_result(video_path, "Dependencies not available")
        
        try:
            # Step 1: Extract audio from video
            print(f"[VideoSentimentModel] Extracting audio from video...")
            audio = self._video_processor.extract_audio_from_video(video_path)
            
            if audio is None:
                return self._create_error_result(video_path, "Audio extraction failed")
            
            # Step 2: Get video information
            video_info = self._video_processor.get_video_info(video_path)
            
            # Step 3: Process audio through sentiment model
            print(f"[VideoSentimentModel] Processing audio through sentiment model...")
            
            # Convert audio to text simulation (in real scenario, use speech-to-text)
            audio_text = self._audio_to_text_simulation(audio, video_path)
            
            # Get sentiment from text
            sentiment_result = self._audio_model.predict(audio_text)
            
            # Add video-specific metadata
            result = {
                **sentiment_result,
                "input_type": "video",
                "video_path": video_path,
                "video_info": video_info,
                "audio_samples": len(audio),
                "audio_duration": len(audio) / self._video_processor.sample_rate,
                "extracted_text": audio_text,
                "processing_pipeline": ["video_load", "audio_extraction", "sentiment_analysis"]
            }
            
            print(f"[VideoSentimentModel] Returning: {result}")
            return result
            
        except Exception as e:
            print(f"[VideoSentimentModel] Error during prediction: {e}")
            return self._create_error_result(video_path, str(e))
    
    def batch_predict(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple video files
        
        Args:
            video_paths (List[str]): List of video file paths
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        print(f"[VideoSentimentModel] Received batch of {len(video_paths)} videos")
        
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                print(f"[VideoSentimentModel] Error processing {video_path}: {e}")
                results.append(self._create_error_result(video_path, str(e)))
        
        print(f"[VideoSentimentModel] Returning batch results: {results}")
        return results
    
    def _audio_to_text_simulation(self, audio: np.ndarray, video_path: str) -> str:
        """
        Simulate speech-to-text conversion from audio
        
        Args:
            audio (np.ndarray): Audio samples
            video_path (str): Original video path (for context)
            
        Returns:
            str: Simulated transcribed text
        """
        # In a real scenario, this would use a speech-to-text model
        # For now, we simulate based on audio characteristics
        
        duration = len(audio) / self._video_processor.sample_rate
        rms = np.sqrt(np.mean(audio**2))
        
        print(f"[VideoSentimentModel] Simulating speech-to-text for {duration:.2f}s audio")
        
        # Simulate different text based on audio characteristics
        if duration < 2.0:
            return "Short video, unclear speech"
        elif rms < 0.1:
            return "Quiet audio, difficult to transcribe"
        elif rms > 0.8:
            return "Loud audio, possible excitement or anger"
        else:
            # Simulate positive/negative text based on video filename
            if any(word in video_path.lower() for word in ["happy", "good", "positive", "love"]):
                return "I love this video! It's amazing and wonderful!"
            elif any(word in video_path.lower() for word in ["sad", "bad", "negative", "hate"]):
                return "This video is terrible, I hate it!"
            else:
                return "This is a neutral video with standard content"
    
    def _create_error_result(self, video_path: str, error_message: str) -> Dict[str, Any]:
        """Create error result when processing fails"""
        return {
            "sentiment": "error",
            "confidence": 0.0,
            "text": f"Error processing video: {error_message}",
            "input_type": "video",
            "video_path": video_path,
            "error": error_message,
            "processing_pipeline": ["error"]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "video_sentiment",
            "version": self.version,
            "confidence_threshold": self.confidence_threshold,
            "sentiment_labels": self.sentiment_labels,
            "supports_video": True,
            "audio_extraction": True,
            "speech_to_text": "simulated"
        }


def test_video_sentiment_model():
    """Test the video sentiment model"""
    print("🎬 TESTING VIDEO SENTIMENT MODEL")
    print("=" * 60)
    
    # Create test configuration
    config = {
        "type": "video_sentiment",
        "version": "v1.0",
        "confidence_threshold": 0.5,
        "sentiment_labels": ["positive", "negative", "neutral"],
        "config": {
            "sample_rate": 16000,
            "max_duration": 30.0,
            "supported_formats": ["mp4", "avi", "mov", "mkv"]
        }
    }
    
    # Initialize model
    model = VideoSentimentModel(config)
    
    # Test with dummy video
    from utils.video_utils import VideoProcessor
    video_processor = VideoProcessor()
    
    print("\n📹 Creating test video...")
    test_video_path = video_processor.create_dummy_video(duration=3.0)
    
    if test_video_path:
        try:
            # Test single prediction
            print(f"\n🔍 Testing single video prediction...")
            result = model.predict(test_video_path)
            print(f"Result: {result}")
            
            # Test batch prediction
            print(f"\n🔍 Testing batch video prediction...")
            batch_results = model.batch_predict([test_video_path, test_video_path])
            print(f"Batch results: {batch_results}")
            
            # Clean up
            video_processor.cleanup_temp_files([test_video_path])
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
    else:
        print("❌ Could not create test video")
    
    print("\n✅ Video sentiment model test completed!")


if __name__ == "__main__":
    test_video_sentiment_model() 