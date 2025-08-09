"""
Postprocessing Module
Handles result formatting and output generation
"""

from typing import Dict, Any, List, Union
import json
from datetime import datetime


class ResultPostprocessor:
    """Postprocess sentiment analysis results"""
    
    def __init__(self):
        self.output_format = "json"
        print("[Postprocessor] Initialized with JSON output format")
    
    def format_single_result(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single prediction result
        
        Args:
            prediction (Dict[str, Any]): Raw prediction from inference
            
        Returns:
            Dict containing formatted result
        """
        print(f"[Postprocessor] Received prediction: {prediction}")
        
        # Format the result
        formatted_result = {
            "input_text": prediction.get("text", ""),
            "sentiment": prediction.get("sentiment", "unknown"),
            "confidence": prediction.get("confidence", 0.0),
            "processed_at": datetime.now().isoformat(),
            "model_info": {
                "version": prediction.get("model_version", "unknown"),
                "inference_timestamp": prediction.get("inference_timestamp", "")
            }
        }
        
        print(f"[Postprocessor] Returning formatted result: {formatted_result}")
        return formatted_result
    
    def format_batch_results(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format batch prediction results
        
        Args:
            predictions (List[Dict[str, Any]]): List of raw predictions
            
        Returns:
            List of formatted results
        """
        print(f"[Postprocessor] Received {len(predictions)} predictions for formatting")
        
        formatted_results = []
        for i, pred in enumerate(predictions):
            formatted = self.format_single_result(pred)
            formatted["batch_index"] = i
            formatted_results.append(formatted)
        
        print(f"[Postprocessor] Returning {len(formatted_results)} formatted results")
        return formatted_results
    
    def add_sentiment_emoji(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add emoji to sentiment result
        
        Args:
            result (Dict[str, Any]): Formatted result
            
        Returns:
            Dict with added emoji
        """
        print(f"[Postprocessor] Adding emoji to result: {result}")
        
        emoji_map = {
            "positive": "😊",
            "negative": "😞", 
            "neutral": "😐",
            "uncertain": "🤔"
        }
        
        sentiment = result.get("sentiment", "unknown")
        emoji = emoji_map.get(sentiment, "❓")
        
        result["sentiment_emoji"] = emoji
        
        print(f"[Postprocessor] Returning result with emoji: {result}")
        return result
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from batch results
        
        Args:
            results (List[Dict[str, Any]]): List of formatted results
            
        Returns:
            Dict containing summary statistics
        """
        print(f"[Postprocessor] Generating summary for {len(results)} results")
        
        sentiment_counts = {}
        total_confidence = 0.0
        
        for result in results:
            sentiment = result.get("sentiment", "unknown")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += result.get("confidence", 0.0)
        
        summary = {
            "total_processed": len(results),
            "sentiment_distribution": sentiment_counts,
            "average_confidence": round(total_confidence / len(results), 3) if results else 0.0,
            "summary_generated_at": datetime.now().isoformat()
        }
        
        print(f"[Postprocessor] Returning summary: {summary}")
        return summary
    
    def save_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                   filename: str = "sentiment_results.json") -> str:
        """
        Save results to JSON file
        
        Args:
            results: Single result or list of results
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        print(f"[Postprocessor] Saving results to {filename}")
        
        if isinstance(results, list):
            output_data = {
                "results": results,
                "summary": self.generate_summary(results),
                "exported_at": datetime.now().isoformat()
            }
        else:
            output_data = {
                "result": results,
                "exported_at": datetime.now().isoformat()
            }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[Postprocessor] Results saved to {filename}")
        return filename 