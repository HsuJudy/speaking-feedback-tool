"""
Sentiment Analysis Model Module
Handles the core sentiment analysis logic
"""

import random
from typing import Dict, Any


class SentimentModel:
    """Dummy sentiment analysis model"""
    
    def __init__(self):
        self.sentiment_labels = ["positive", "negative", "neutral"]
        self.confidence_threshold = 0.5
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for given text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing sentiment label and confidence score
        """
        print(f"[SentimentModel] Received text: '{text}'")
        
        # Dummy prediction logic
        if any(word in text.lower() for word in ["love", "amazing", "perfect", "delicious", "great"]):
            sentiment = "positive"
            confidence = random.uniform(0.7, 0.95)
        elif any(word in text.lower() for word in ["worst", "terrible", "bad", "hate", "awful"]):
            sentiment = "negative"
            confidence = random.uniform(0.7, 0.95)
        else:
            sentiment = "neutral"
            confidence = random.uniform(0.4, 0.6)
        
        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "text": text
        }
        
        print(f"[SentimentModel] Returning: {result}")
        return result
    
    def batch_predict(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            List of prediction results
        """
        print(f"[SentimentModel] Received batch of {len(texts)} texts")
        results = [self.predict(text) for text in texts]
        print(f"[SentimentModel] Returning batch results: {results}")
        return results 