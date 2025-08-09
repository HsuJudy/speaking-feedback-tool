"""
Text Preprocessing Module
Handles text cleaning and preparation for sentiment analysis
"""

import re
from typing import Union, List


class TextPreprocessor:
    """Text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        print(f"[Preprocessor] Received text: '{text}'")
        
        # Convert to lowercase
        cleaned = text.lower()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep punctuation for sentiment
        cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,]', '', cleaned)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        print(f"[Preprocessor] Returning cleaned text: '{cleaned}'")
        return cleaned
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove common stop words
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stop words removed
        """
        print(f"[Preprocessor] Removing stop words from: '{text}'")
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        result = ' '.join(filtered_words)
        
        print(f"[Preprocessor] Returning text without stop words: '{result}'")
        return result
    
    def preprocess(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Main preprocessing function
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Preprocessed text(s)
        """
        if isinstance(text, list):
            print(f"[Preprocessor] Received list of {len(text)} texts")
            results = [self.clean_text(t) for t in text]
            print(f"[Preprocessor] Returning list of preprocessed texts: {results}")
            return results
        else:
            return self.clean_text(text) 