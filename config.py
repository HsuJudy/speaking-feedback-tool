"""
Configuration Module for Sentiment Analysis Pipeline
Centralized configuration management
"""

import os
from typing import Dict, Any


class PipelineConfig:
    """Configuration class for the sentiment analysis pipeline"""
    
    def __init__(self):
        # Model configuration
        self.model_config = {
            "model_type": "dummy",
            "version": "v1.0",
            "confidence_threshold": 0.5,
            "sentiment_labels": ["positive", "negative", "neutral", "uncertain"]
        }
        
        # Preprocessing configuration
        self.preprocessing_config = {
            "remove_stop_words": False,
            "lowercase": True,
            "remove_special_chars": True,
            "normalize_whitespace": True
        }
        
        # Inference configuration
        self.inference_config = {
            "batch_size": 32,
            "enable_metadata": True,
            "add_timestamps": True
        }
        
        # Postprocessing configuration
        self.postprocessing_config = {
            "output_format": "json",
            "include_emoji": True,
            "generate_summary": True,
            "save_results": True
        }
        
        # File paths
        self.paths = {
            "data_dir": "data",
            "models_dir": "models",
            "pipeline_dir": "pipeline",
            "output_dir": "output",
            "sample_inputs": "data/sample_inputs.json"
        }
        
        # Logging configuration
        self.logging_config = {
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "enable_console_logging": True,
            "enable_file_logging": False,
            "log_file": "pipeline.log"
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_config.copy()
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self.preprocessing_config.copy()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.inference_config.copy()
    
    def get_postprocessing_config(self) -> Dict[str, Any]:
        """Get postprocessing configuration"""
        return self.postprocessing_config.copy()
    
    def get_paths(self) -> Dict[str, str]:
        """Get file paths configuration"""
        return self.paths.copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.logging_config.copy()
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value
        
        Args:
            section (str): Configuration section (model_config, preprocessing_config, etc.)
            key (str): Configuration key
            value (Any): New value
        """
        if hasattr(self, section):
            config_dict = getattr(self, section)
            if isinstance(config_dict, dict):
                config_dict[key] = value
                print(f"[Config] Updated {section}.{key} = {value}")
            else:
                print(f"[Config] Error: {section} is not a dictionary")
        else:
            print(f"[Config] Error: Unknown configuration section '{section}'")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        print("[Config] Loading configuration from environment variables...")
        
        # Model config from env
        if os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD"):
            threshold = float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD"))
            self.update_config("model_config", "confidence_threshold", threshold)
        
        # Logging config from env
        if os.getenv("LOG_LEVEL"):
            self.update_config("logging_config", "log_level", os.getenv("LOG_LEVEL"))
        
        print("[Config] Environment configuration loaded")
    
    def print_config(self) -> None:
        """Print current configuration"""
        print("\n" + "=" * 50)
        print("PIPELINE CONFIGURATION")
        print("=" * 50)
        
        print(f"\nModel Config: {self.model_config}")
        print(f"Preprocessing Config: {self.preprocessing_config}")
        print(f"Inference Config: {self.inference_config}")
        print(f"Postprocessing Config: {self.postprocessing_config}")
        print(f"Paths: {self.paths}")
        print(f"Logging Config: {self.logging_config}")
        print("=" * 50)


# Global configuration instance
config = PipelineConfig() 
config.print_config()