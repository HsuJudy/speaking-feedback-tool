"""
Test script for video processing pipeline
Demonstrates video → audio extraction → sentiment analysis
"""

import json
import os
import tempfile
import numpy as np
from models.model_factory import ModelFactory
from utils.video_utils import VideoProcessor


def test_video_processing_pipeline():
    """Test the complete video processing pipeline"""
    print("🎬 TESTING VIDEO PROCESSING PIPELINE")
    print("=" * 60)
    
    # Initialize components
    factory = ModelFactory("config/models.yaml")
    video_processor = VideoProcessor()
    
    # Test different scenarios
    test_scenarios = [
        {
            "name": "Short Video (2s)",
            "duration": 2.0,
            "expected_issue": "Short duration may cause unclear speech"
        },
        {
            "name": "Normal Video (5s)",
            "duration": 5.0,
            "expected_issue": "None"
        },
        {
            "name": "Long Video (15s)",
            "duration": 15.0,
            "expected_issue": "May be trimmed to max duration"
        }
    ]
    
    temp_files = []
    
    try:
        for scenario in test_scenarios:
            print(f"\n{'='*20} TESTING: {scenario['name']} {'='*20}")
            
            # Create test video
            print(f"📹 Creating {scenario['duration']}s video...")
            video_path = video_processor.create_dummy_video(duration=scenario['duration'])
            
            if video_path:
                temp_files.append(video_path)
                
                # Get video info
                video_info = video_processor.get_video_info(video_path)
                print(f"Video info: {video_info}")
                
                # Test audio extraction
                print(f"🎵 Extracting audio...")
                audio = video_processor.extract_audio_from_video(video_path)
                
                if audio is not None:
                    print(f"✅ Audio extracted: {len(audio)} samples")
                    print(f"   Duration: {len(audio) / video_processor.sample_rate:.2f}s")
                    print(f"   RMS level: {np.sqrt(np.mean(audio**2)):.4f}")
                    
                    # Test complete pipeline
                    print(f"🔄 Testing complete pipeline...")
                    processed_audio = video_processor.video_to_audio_pipeline(video_path)
                    
                    if processed_audio is not None:
                        print(f"✅ Pipeline completed: {len(processed_audio)} samples")
                        
                        # Test with video sentiment model
                        print(f"🧠 Testing with video sentiment model...")
                        video_model = factory.get_model("video_sentiment")
                        result = video_model.predict(video_path)
                        
                        print(f"Video sentiment result:")
                        print(json.dumps(result, indent=2))
                        
                    else:
                        print(f"❌ Pipeline failed")
                else:
                    print(f"❌ Audio extraction failed")
            else:
                print(f"❌ Video creation failed")
    
    finally:
        # Clean up temporary files
        print(f"\n🧹 Cleaning up {len(temp_files)} temporary files...")
        video_processor.cleanup_temp_files(temp_files)


def test_failure_scenarios():
    """Test various failure scenarios"""
    print("\n🚨 TESTING FAILURE SCENARIOS")
    print("=" * 60)
    
    factory = ModelFactory("config/models.yaml")
    video_model = factory.get_model("video_sentiment")
    
    failure_scenarios = [
        {
            "name": "Non-existent video file",
            "path": "nonexistent_video.mp4",
            "expected_error": "File not found"
        },
        {
            "name": "Invalid video format",
            "path": "test.txt",
            "expected_error": "Invalid format"
        },
        {
            "name": "Corrupted video file",
            "path": "corrupted_video.mp4",
            "expected_error": "Corrupted file"
        }
    ]
    
    for scenario in failure_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")
        
        # Create a dummy file for invalid format test
        if scenario['name'] == "Invalid video format":
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                f.write(b"This is not a video file")
                scenario['path'] = f.name
        
        result = video_model.predict(scenario['path'])
        
        print(f"Result: {result.get('sentiment', 'unknown')}")
        print(f"Error: {result.get('error', 'none')}")
        
        # Clean up
        if scenario['name'] == "Invalid video format":
            os.unlink(scenario['path'])


def test_batch_video_processing():
    """Test batch processing of multiple videos"""
    print("\n📦 TESTING BATCH VIDEO PROCESSING")
    print("=" * 60)
    
    factory = ModelFactory("config/models.yaml")
    video_model = factory.get_model("video_sentiment")
    video_processor = VideoProcessor()
    
    # Create multiple test videos
    video_paths = []
    temp_files = []
    
    try:
        for i in range(3):
            video_path = video_processor.create_dummy_video(duration=3.0 + i)
            if video_path:
                video_paths.append(video_path)
                temp_files.append(video_path)
        
        if video_paths:
            print(f"Created {len(video_paths)} test videos")
            
            # Test batch processing
            print(f"🔄 Processing {len(video_paths)} videos in batch...")
            batch_results = video_model.batch_predict(video_paths)
            
            print(f"Batch processing results:")
            for i, result in enumerate(batch_results):
                print(f"Video {i+1}: {result.get('sentiment', 'error')} "
                      f"(confidence: {result.get('confidence', 0):.3f})")
            
            # Generate summary
            sentiments = [r.get('sentiment') for r in batch_results]
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            error_count = sentiments.count('error')
            
            print(f"\n📊 Batch Summary:")
            print(f"  Positive: {positive_count}")
            print(f"  Negative: {negative_count}")
            print(f"  Neutral: {neutral_count}")
            print(f"  Errors: {error_count}")
        
    finally:
        # Clean up
        video_processor.cleanup_temp_files(temp_files)


def test_performance_metrics():
    """Test performance metrics for video processing"""
    print("\n⚡ TESTING PERFORMANCE METRICS")
    print("=" * 60)
    
    import time
    
    factory = ModelFactory("config/models.yaml")
    video_model = factory.get_model("video_sentiment")
    video_processor = VideoProcessor()
    
    # Create test video
    video_path = video_processor.create_dummy_video(duration=5.0)
    
    if video_path:
        try:
            # Measure processing time
            print(f"⏱️ Measuring processing time...")
            
            start_time = time.time()
            result = video_model.predict(video_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            video_info = result.get('video_info', {})
            duration = video_info.get('duration', 0)
            
            print(f"📊 Performance Metrics:")
            print(f"  Video duration: {duration:.2f}s")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Processing ratio: {processing_time/duration:.2f}x real-time")
            print(f"  Audio samples: {result.get('audio_samples', 0)}")
            print(f"  Audio duration: {result.get('audio_duration', 0):.2f}s")
            
            # Calculate efficiency
            if duration > 0:
                efficiency = duration / processing_time
                print(f"  Efficiency: {efficiency:.2f}x (higher is better)")
            
        finally:
            video_processor.cleanup_temp_files([video_path])


def main():
    """Main test function"""
    print("🎭 COMPREHENSIVE VIDEO PROCESSING TEST")
    print("=" * 60)
    
    # Test basic pipeline
    test_video_processing_pipeline()
    
    # Test failure scenarios
    test_failure_scenarios()
    
    # Test batch processing
    test_batch_video_processing()
    
    # Test performance
    test_performance_metrics()
    
    print("\n✅ All video processing tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 