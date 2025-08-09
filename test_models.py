"""
Test script for dynamic model loading
Demonstrates how to use the ModelFactory to load different models
"""

import json
from models.model_factory import ModelFactory


def test_model_factory():
    """Test the model factory with different model configurations"""
    print("🧪 TESTING MODEL FACTORY")
    print("=" * 50)
    
    # Initialize model factory
    factory = ModelFactory("config/models.yaml")
    
    # List available models
    print(f"\n📋 Available models: {factory.list_available_models()}")
    
    # Test each model
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it!",
        "The movie was okay, nothing special."
    ]
    
    for model_name in factory.list_available_models():
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
    
    print("\n✅ Model factory testing completed!")


def test_model_switching():
    """Test switching between different models"""
    print("\n🔄 TESTING MODEL SWITCHING")
    print("=" * 50)
    
    factory = ModelFactory("config/models.yaml")
    test_text = "This restaurant has the most delicious food I've ever tasted!"
    
    print(f"Test text: '{test_text}'")
    
    for model_name in ["dummy", "huggingface", "nemo"]:
        try:
            print(f"\n--- Testing {model_name} model ---")
            model = factory.get_model(model_name)
            result = model.predict(test_text)
            print(f"Model: {model_name}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue


if __name__ == "__main__":
    test_model_factory()
    test_model_switching() 