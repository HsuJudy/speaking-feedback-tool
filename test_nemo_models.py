"""
Test script for NeMo QuartzNet and ConformerCTC models
Demonstrates speech recognition models with sentiment analysis
"""

import json
import os
from models.model_factory import ModelFactory


def test_nemo_speech_models():
    """Test the NeMo speech recognition models"""
    print("🎤 TESTING NEMO SPEECH RECOGNITION MODELS")
    print("=" * 60)
    
    # Initialize model factory
    factory = ModelFactory("config/models.yaml")
    
    # Test speech-related models
    speech_models = ["quartznet15x5", "conformer_ctc"]
    
    # Sample texts that simulate speech input
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service!",
        "The movie was okay, nothing special but not bad either.",
        "I'm feeling neutral about this situation.",
        "This restaurant has the most delicious food I've ever tasted!"
    ]
    
    for model_name in speech_models:
        print(f"\n{'='*20} TESTING {model_name.upper()} {'='*20}")
        
        try:
            # Get model info
            model_info = factory.get_model_info(model_name)
            print(f"Model info: {json.dumps(model_info, indent=2)}")
            
            # Load model
            model = factory.get_model(model_name)
            print(f"Model loaded: {model.get_model_info()}")
            
            # Test single prediction
            print(f"\n🔍 Testing single prediction:")
            result = model.predict(test_texts[0])
            print(f"Input: '{test_texts[0]}'")
            print(f"Output: {json.dumps(result, indent=2)}")
            
            # Test batch prediction
            print(f"\n🔍 Testing batch prediction:")
            batch_results = model.batch_predict(test_texts)
            for i, (text, result) in enumerate(zip(test_texts, batch_results)):
                print(f"Text {i+1}: '{text}'")
                print(f"Result: {json.dumps(result, indent=2)}")
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            continue
    
    print("\n✅ NeMo speech model testing completed!")


def test_audio_simulation():
    """Simulate audio processing with speech models"""
    print("\n🎵 SIMULATING AUDIO PROCESSING")
    print("=" * 60)
    
    factory = ModelFactory("config/models.yaml")
    
    # Simulate different audio scenarios
    audio_scenarios = [
        {
            "description": "Positive customer feedback",
            "text": "I absolutely love this product! It's fantastic and works perfectly.",
            "expected_sentiment": "positive"
        },
        {
            "description": "Negative customer complaint", 
            "text": "This is the worst service I've ever experienced. Terrible!",
            "expected_sentiment": "negative"
        },
        {
            "description": "Neutral feedback",
            "text": "The product is okay, nothing special but not bad either.",
            "expected_sentiment": "neutral"
        }
    ]
    
    for scenario in audio_scenarios:
        print(f"\n📝 Scenario: {scenario['description']}")
        print(f"Simulated audio transcript: '{scenario['text']}'")
        print(f"Expected sentiment: {scenario['expected_sentiment']}")
        
        # Test with both speech models
        for model_name in ["quartznet15x5", "conformer_ctc"]:
            try:
                model = factory.get_model(model_name)
                result = model.predict(scenario['text'])
                
                print(f"  {model_name}: {result['sentiment']} (confidence: {result['confidence']})")
                print(f"    Model type: {result.get('model_type', 'unknown')}")
                print(f"    Audio processed: {result.get('audio_processed', False)}")
                print(f"    Sample rate: {result.get('sample_rate', 'N/A')}")
                
            except Exception as e:
                print(f"  {model_name}: Error - {e}")


def test_model_comparison():
    """Compare different NeMo models"""
    print("\n📊 MODEL COMPARISON")
    print("=" * 60)
    
    factory = ModelFactory("config/models.yaml")
    test_text = "This restaurant has the most delicious food I've ever tasted!"
    
    print(f"Test text: '{test_text}'")
    print("\nModel Comparison:")
    print("-" * 60)
    
    models_to_compare = ["dummy", "nemo", "quartznet15x5", "conformer_ctc"]
    
    for model_name in models_to_compare:
        try:
            model = factory.get_model(model_name)
            result = model.predict(test_text)
            
            print(f"{model_name:15} | {result['sentiment']:10} | {result['confidence']:8.3f} | {result.get('model_type', 'N/A')}")
            
        except Exception as e:
            print(f"{model_name:15} | ERROR     | {str(e)[:20]}...")


def main():
    """Main function to test NeMo speech models"""
    print("🎭 NEMO SPEECH RECOGNITION MODEL TESTING")
    print("=" * 60)
    
    # Test basic functionality
    test_nemo_speech_models()
    
    # Test audio simulation
    test_audio_simulation()
    
    # Compare models
    test_model_comparison()
    
    print("\n✅ All NeMo model tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 