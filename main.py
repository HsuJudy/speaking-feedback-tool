"""
Main Pipeline for Sentiment Analysis
Demonstrates the complete MLOps-style pipeline
"""

import json
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocessor import TextPreprocessor
from pipeline.inference import InferenceEngine
from pipeline.postprocessor import ResultPostprocessor


def run_single_pipeline(text: str, model_name: str = None, use_wandb: bool = False) -> dict:
    """
    Run a single text through the complete pipeline
    
    Args:
        text (str): Input text to analyze
        model_name (str, optional): Name of the model to use
        use_wandb (bool): Whether to enable W&B tracking
        
    Returns:
        dict: Final processed result
    """
    print("=" * 60)
    print(f"STARTING SINGLE TEXT PIPELINE (Model: {model_name or 'default'})")
    print("=" * 60)
    
    # Initialize pipeline components
    print("\n[Main] Initializing pipeline components...")
    preprocessor = TextPreprocessor()
    inference_engine = InferenceEngine(model_name=model_name, use_wandb=use_wandb)
    postprocessor = ResultPostprocessor()
    
    # Step 1: Preprocessing
    print("\n[Main] Step 1: Preprocessing")
    cleaned_text = preprocessor.preprocess(text)
    
    # Step 2: Inference
    print("\n[Main] Step 2: Inference")
    raw_prediction = inference_engine.predict_single(cleaned_text)
    
    # Step 3: Postprocessing
    print("\n[Main] Step 3: Postprocessing")
    formatted_result = postprocessor.format_single_result(raw_prediction)
    final_result = postprocessor.add_sentiment_emoji(formatted_result)
    
    print("\n[Main] Pipeline completed successfully!")
    print("=" * 60)
    
    return final_result


def run_batch_pipeline(texts: list, model_name: str = None, use_wandb: bool = False) -> list:
    """
    Run multiple texts through the complete pipeline
    
    Args:
        texts (list): List of input texts
        model_name (str, optional): Name of the model to use
        use_wandb (bool): Whether to enable W&B tracking
        
    Returns:
        list: List of final processed results
    """
    print("=" * 60)
    print(f"STARTING BATCH PIPELINE (Model: {model_name or 'default'})")
    print("=" * 60)
    
    # Initialize pipeline components
    print("\n[Main] Initializing pipeline components...")
    preprocessor = TextPreprocessor()
    inference_engine = InferenceEngine(model_name=model_name, use_wandb=use_wandb)
    postprocessor = ResultPostprocessor()
    
    # Step 1: Preprocessing
    print("\n[Main] Step 1: Preprocessing")
    cleaned_texts = preprocessor.preprocess(texts)
    
    # Step 2: Inference
    print("\n[Main] Step 2: Inference")
    raw_predictions = inference_engine.predict_batch(cleaned_texts)
    
    # Step 3: Postprocessing
    print("\n[Main] Step 3: Postprocessing")
    formatted_results = postprocessor.format_batch_results(raw_predictions)
    final_results = [postprocessor.add_sentiment_emoji(result) for result in formatted_results]
    
    print("\n[Main] Batch pipeline completed successfully!")
    print("=" * 60)
    
    return final_results


def load_sample_data() -> list:
    """Load sample sentences from data file"""
    try:
        with open('data/sample_inputs.json', 'r') as f:
            data = json.load(f)
            return data.get('sample_sentences', [])
    except FileNotFoundError:
        print("[Main] Sample data file not found, using default sentences")
        return [
            "I love this product! It's amazing and works perfectly.",
            "This is the worst experience I've ever had. Terrible service!",
            "The movie was okay, nothing special but not bad either."
        ]


def main():
    """Main function to demonstrate the pipeline"""
    print("🎭 VIBE-CHECK SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Check for W&B flag
    use_wandb = "--wandb" in sys.argv
    if use_wandb:
        print("🔮 W&B tracking enabled")
    
    # Load sample data
    sample_sentences = load_sample_data()
    
    # Test different models
    models_to_test = ["dummy", "huggingface", "nemo", "quartznet15x5", "conformer_ctc", "video_sentiment", "audio_emotion"]
    
    for model_name in models_to_test:
        print(f"\n{'='*20} TESTING {model_name.upper()} MODEL {'='*20}")
        
        try:
            # Run single text pipeline
            print(f"\n🔍 RUNNING SINGLE TEXT ANALYSIS WITH {model_name.upper()}")
            sample_text = sample_sentences[0]
            single_result = run_single_pipeline(sample_text, model_name, use_wandb=use_wandb)
            
            print(f"\n📊 SINGLE RESULT ({model_name}):")
            print(json.dumps(single_result, indent=2))
            
            # Run batch pipeline
            print(f"\n🔍 RUNNING BATCH ANALYSIS WITH {model_name.upper()}")
            batch_results = run_batch_pipeline(sample_sentences, model_name, use_wandb=use_wandb)
            
            print(f"\n📊 BATCH RESULTS ({model_name}):")
            for i, result in enumerate(batch_results):
                print(f"\nResult {i+1}:")
                print(json.dumps(result, indent=2))
            
            # Generate and display summary
            print(f"\n📈 GENERATING SUMMARY FOR {model_name.upper()}")
            postprocessor = ResultPostprocessor()
            summary = postprocessor.generate_summary(batch_results)
            
            print(f"\n📊 SUMMARY ({model_name}):")
            print(json.dumps(summary, indent=2))
            
            # Save results
            print(f"\n💾 SAVING RESULTS FOR {model_name.upper()}")
            postprocessor.save_results(batch_results, f"pipeline_results_{model_name}.json")
            
        except Exception as e:
            print(f"❌ Error testing {model_name} model: {e}")
            continue
    
    print("\n✅ Pipeline demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 