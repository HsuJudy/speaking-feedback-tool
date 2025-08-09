"""
Video processing utilities for extracting audio from video files
Handles video loading, audio extraction, and format conversion
"""

import os
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any
import warnings

try:
    import cv2
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    warnings.warn("OpenCV not available. Install opencv-python for video support.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install numpy for array operations.")

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    warnings.warn("Audio processing libraries not available. Install librosa and soundfile for audio support.")

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("Can't check for GPU — PyTorch not available.")



# Device detection for GPU support
device = "cuda" if GPU_AVAILABLE else "cpu"
print(f"Using device: {device}")

class VideoProcessor:
    """Video processing utilities for audio extraction"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        
        if not VIDEO_AVAILABLE:
            print("[VideoProcessor] Warning: Video libraries not available")
        if not AUDIO_AVAILABLE:
            print("[VideoProcessor] Warning: Audio libraries not available")
    
    def extract_audio_from_video(self, video_path: str):
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Audio samples at target sample rate (numpy array if available, else None)
        """
        if not NUMPY_AVAILABLE:
            print("[VideoProcessor] Error: NumPy not available for audio processing")
            return None
            
        if not os.path.exists(video_path):
            print(f"[VideoProcessor] Error: Video file {video_path} not found")
            return None
        
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                temp_audio_path
            ]
            
            print(f"[VideoProcessor] Extracting audio from {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[VideoProcessor] FFmpeg error: {result.stderr}")
                return None
            
            # Load the extracted audio
            if AUDIO_AVAILABLE:
                audio, sr = librosa.load(temp_audio_path, sr=self.sample_rate)
                
                # Clean up temporary file
                os.unlink(temp_audio_path)
                
                # Trim to max duration
                if len(audio) > self.max_samples:
                    audio = audio[:self.max_samples]
                    print(f"[VideoProcessor] Trimmed audio to {self.max_duration}s")
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                print(f"[VideoProcessor] Extracted audio: {len(audio)} samples at {sr}Hz")
                return audio
            else:
                print("[VideoProcessor] Audio libraries not available")
                return None
                
        except Exception as e:
            print(f"[VideoProcessor] Error extracting audio: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about video file
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Dict: Video information
        """
        if not VIDEO_AVAILABLE:
            return {"error": "OpenCV not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": round(duration, 2),
                "resolution": f"{width}x{height}"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def create_dummy_video(self, duration: float = 5.0, resolution: Tuple[int, int] = (640, 480)) -> str:
        """
        Create a dummy video file for testing
        
        Args:
            duration (float): Duration in seconds
            resolution (tuple): Video resolution (width, height)
            
        Returns:
            str: Path to created video file
        """
        if not VIDEO_AVAILABLE:
            print("[VideoProcessor] Cannot create dummy video: OpenCV not available")
            return None
        
        try:
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name
            
            # Video properties
            fps = 30
            width, height = resolution
            frame_count = int(duration * fps)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            # Generate frames
            for i in range(frame_count):
                # Create a simple animated frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add some visual elements
                cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {i+1}", (width//2-50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            
            print(f"[VideoProcessor] Created dummy video: {temp_video_path}")
            return temp_video_path
            
        except Exception as e:
            print(f"[VideoProcessor] Error creating dummy video: {e}")
            return None
    
    def video_to_audio_pipeline(self, video_path: str):
        """
        Complete pipeline: video → audio extraction → preprocessing
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Processed audio samples (numpy array if available, else None)
        """
        print(f"[VideoProcessor] Starting video-to-audio pipeline for {video_path}")
        
        # Step 1: Get video info
        video_info = self.get_video_info(video_path)
        print(f"[VideoProcessor] Video info: {video_info}")
        
        # Step 2: Extract audio
        audio = self.extract_audio_from_video(video_path)
        if audio is None:
            print("[VideoProcessor] Failed to extract audio")
            return None
        
        # Step 3: Preprocess audio (if audio libraries available)
        if AUDIO_AVAILABLE:
            from utils.audio_utils import AudioProcessor
            audio_processor = AudioProcessor(self.sample_rate, self.max_duration)
            processed_audio = audio_processor.preprocess_audio(audio)
            print(f"[VideoProcessor] Audio preprocessing completed: {len(processed_audio)} samples")
            return processed_audio
        else:
            print("[VideoProcessor] Audio preprocessing skipped (libraries not available)")
            return audio
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"[VideoProcessor] Cleaned up: {file_path}")
            except Exception as e:
                print(f"[VideoProcessor] Error cleaning up {file_path}: {e}")


def test_video_processor():
    """Test the video processor"""
    print("🎬 TESTING VIDEO PROCESSOR")
    print("=" * 50)
    
    processor = VideoProcessor()
    
    # Test dummy video creation
    print("\n📹 Creating dummy video...")
    dummy_video_path = processor.create_dummy_video(duration=3.0)
    
    if dummy_video_path:
        # Get video info
        video_info = processor.get_video_info(dummy_video_path)
        print(f"Video info: {video_info}")
        
        # Test audio extraction
        print("\n🎵 Extracting audio from video...")
        audio = processor.extract_audio_from_video(dummy_video_path)
        
        if audio is not None:
            print(f"Extracted audio length: {len(audio)} samples")
            print(f"Audio duration: {len(audio) / processor.sample_rate:.2f}s")
            
            # Test complete pipeline
            print("\n🔄 Testing complete pipeline...")
            processed_audio = processor.video_to_audio_pipeline(dummy_video_path)
            
            if processed_audio is not None:
                print(f"Pipeline completed: {len(processed_audio)} samples")
            
            # Clean up
            processor.cleanup_temp_files([dummy_video_path])
        else:
            print("❌ Audio extraction failed")
    else:
        print("❌ Dummy video creation failed")
    
    print("\n✅ Video processor test completed!")


if __name__ == "__main__":
    test_video_processor() 