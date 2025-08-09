"""
Inference Module
Handles model inference and prediction orchestration
"""

from typing import Dict, Any, List, Union
from models.model_factory import ModelFactory


class InferenceEngine:
    """Inference engine for sentiment analysis"""
    
    def __init__(self, model_name: str = None, config_path: str = "config/models.yaml", use_wandb: bool = False):
        self.model_factory = ModelFactory(config_path, use_wandb=use_wandb)
        self.model = self.model_factory.get_model(model_name)
        self.model_name = model_name or self.model_factory.config.get("default_model", "dummy")
        self.use_wandb = use_wandb
        self.wandb_manager = self.model_factory.wandb_manager
        print(f"[InferenceEngine] Initialized with {self.model_name} model")
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            Dict containing prediction results
        """
        print(f"[InferenceEngine] Received text for prediction: '{text}'")
        
        # Run inference
        prediction = self.model.predict(text)
        
        # Add metadata
        model_info = self.model.get_model_info()
        result = {
            **prediction,
            "model_version": f"{model_info['type']}_{model_info['version']}",
            "model_name": self.model_name,
            "inference_timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Log to W&B if enabled
        if self.use_wandb and self.wandb_manager:
            try:
                # Log prediction
                self.wandb_manager.log_predictions([result], self.model_name)
                
                # Log metrics
                metrics = {
                    f"{self.model_name}_confidence": result.get("confidence", 0.0),
                    f"{self.model_name}_sentiment": result.get("sentiment", "unknown")
                }
                self.wandb_manager.log_metrics(metrics)
                
            except Exception as e:
                print(f"[InferenceEngine] Warning: Failed to log to W&B: {e}")
        
        print(f"[InferenceEngine] Returning prediction: {result}")
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (List[str]): List of preprocessed texts
            
        Returns:
            List of prediction results
        """
        print(f"[InferenceEngine] Received batch of {len(texts)} texts for prediction")
        
        # Run batch inference
        predictions = self.model.batch_predict(texts)
        
        # Add metadata to each prediction
        model_info = self.model.get_model_info()
        results = []
        for i, pred in enumerate(predictions):
            result = {
                **pred,
                "model_version": f"{model_info['type']}_{model_info['version']}",
                "model_name": self.model_name,
                "inference_timestamp": "2024-01-01T00:00:00Z",
                "batch_index": i
            }
            results.append(result)
        
        # Log to W&B if enabled
        if self.use_wandb and self.wandb_manager:
            try:
                # Log batch predictions
                self.wandb_manager.log_predictions(results, self.model_name)
                
                # Log batch metrics
                confidences = [r.get("confidence", 0.0) for r in results]
                sentiments = [r.get("sentiment", "unknown") for r in results]
                
                metrics = {
                    f"{self.model_name}_batch_size": len(results),
                    f"{self.model_name}_avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                    f"{self.model_name}_positive_count": sentiments.count("positive"),
                    f"{self.model_name}_negative_count": sentiments.count("negative"),
                    f"{self.model_name}_neutral_count": sentiments.count("neutral")
                }
                self.wandb_manager.log_metrics(metrics)
                
            except Exception as e:
                print(f"[InferenceEngine] Warning: Failed to log batch to W&B: {e}")
        
        print(f"[InferenceEngine] Returning batch predictions: {results}")
        return results
    
    def predict_with_confidence_threshold(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict sentiment with confidence threshold filtering
        
        Args:
            text (str): Preprocessed text
            threshold (float): Minimum confidence threshold
            
        Returns:
            Dict containing prediction results with confidence filtering
        """
        print(f"[InferenceEngine] Predicting with confidence threshold {threshold} for: '{text}'")
        
        prediction = self.predict_single(text)
        
        # Apply confidence threshold
        if prediction["confidence"] < threshold:
            prediction["sentiment"] = "uncertain"
            prediction["confidence"] = prediction["confidence"]
        
        print(f"[InferenceEngine] Returning threshold-filtered prediction: {prediction}")
        return prediction 