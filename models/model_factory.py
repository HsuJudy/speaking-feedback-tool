"""
Model Factory for Dynamic Model Loading
Supports loading different model types (Dummy, HuggingFace, NeMo) based on YAML configuration
"""

import yaml
import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .sentiment_model import SentimentModel


class BaseModel(ABC):
    """Abstract base class for all sentiment analysis models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("type", "unknown")
        self.version = config.get("version", "v1.0")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.sentiment_labels = config.get("sentiment_labels", ["positive", "negative", "neutral"])
        
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text"""
        pass
    
    @abstractmethod
    def batch_predict(self, texts: list) -> list:
        """Predict sentiment for multiple texts"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": self.model_type,
            "version": self.version,
            "confidence_threshold": self.confidence_threshold,
            "sentiment_labels": self.sentiment_labels
        }


class DummyModel(BaseModel):
    """Dummy model implementation (existing functionality)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print(f"[DummyModel] Initialized with config: {config}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Dummy prediction logic"""
        print(f"[DummyModel] Received text: '{text}'")
        
        import random
        
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
        
        print(f"[DummyModel] Returning: {result}")
        return result
    
    def batch_predict(self, texts: list) -> list:
        """Batch prediction"""
        print(f"[DummyModel] Received batch of {len(texts)} texts")
        results = [self.predict(text) for text in texts]
        print(f"[DummyModel] Returning batch results: {results}")
        return results


class HuggingFaceModel(BaseModel):
    """HuggingFace Transformers model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._model = None
        self._tokenizer = None
        print(f"[HuggingFaceModel] Initialized with config: {config}")
    
    def _load_model(self):
        """Lazy load the HuggingFace model"""
        if self._model is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                
                model_name = self.config.get("model_name", "cardiffnlp/twitter-roberta-base-sentiment-latest")
                print(f"[HuggingFaceModel] Loading model: {model_name}")
                
                # Create sentiment analysis pipeline
                self._model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=self.model_config.get("device", "auto"),
                    return_all_scores=self.model_config.get("return_all_scores", False)
                )
                
                print(f"[HuggingFaceModel] Model loaded successfully")
                
            except ImportError:
                print("[HuggingFaceModel] Warning: transformers not installed, falling back to dummy model")
                return DummyModel(self.config)
            except Exception as e:
                print(f"[HuggingFaceModel] Error loading model: {e}, falling back to dummy model")
                return DummyModel(self.config)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using HuggingFace model"""
        print(f"[HuggingFaceModel] Received text: '{text}'")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.predict(text)
        
        try:
            # Run prediction
            result = self._model(text)[0]
            
            # Map HuggingFace labels to our standard labels
            label_mapping = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral", 
                "LABEL_2": "positive"
            }
            
            sentiment = label_mapping.get(result["label"], result["label"])
            confidence = result["score"]
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                sentiment = "uncertain"
            
            output = {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "text": text
            }
            
            print(f"[HuggingFaceModel] Returning: {output}")
            return output
            
        except Exception as e:
            print(f"[HuggingFaceModel] Error during prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.predict(text)
    
    def batch_predict(self, texts: list) -> list:
        """Batch prediction"""
        print(f"[HuggingFaceModel] Received batch of {len(texts)} texts")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.batch_predict(texts)
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            print(f"[HuggingFaceModel] Returning batch results: {results}")
            return results
            
        except Exception as e:
            print(f"[HuggingFaceModel] Error during batch prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.batch_predict(texts)


class NeMoModel(BaseModel):
    """NVIDIA NeMo model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._model = None
        print(f"[NeMoModel] Initialized with config: {config}")
    
    def _load_model(self):
        """Lazy load the NeMo model"""
        if self._model is None:
            try:
                import nemo.collections.nlp as nemo_nlp
                from nemo.collections.nlp.models import TextClassificationModel
                
                model_path = self.model_config.get("model_path", "./models/nemo_sentiment_model.nemo")
                print(f"[NeMoModel] Loading model from: {model_path}")
                
                # Load NeMo model
                self._model = TextClassificationModel.restore_from(model_path)
                
                # Set device
                device = self.model_config.get("device", "auto")
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._model = self._model.to(device)
                self._model.eval()
                
                print(f"[NeMoModel] Model loaded successfully on {device}")
                
            except ImportError:
                print("[NeMoModel] Warning: nemo not installed, falling back to dummy model")
                return DummyModel(self.config)
            except Exception as e:
                print(f"[NeMoModel] Error loading model: {e}, falling back to dummy model")
                return DummyModel(self.config)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using NeMo model"""
        print(f"[NeMoModel] Received text: '{text}'")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.predict(text)
        
        try:
            import torch
            
            # Prepare input
            inputs = [text]
            
            # Run inference
            with torch.no_grad():
                predictions = self._model.classifytext(texts=inputs)
            
            # Process results
            logits = predictions[0]  # Assuming single text input
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            # Map class index to sentiment
            sentiment_labels = self.sentiment_labels
            if predicted_class < len(sentiment_labels):
                sentiment = sentiment_labels[predicted_class]
            else:
                sentiment = "unknown"
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                sentiment = "uncertain"
            
            output = {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "text": text
            }
            
            print(f"[NeMoModel] Returning: {output}")
            return output
            
        except Exception as e:
            print(f"[NeMoModel] Error during prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.predict(text)
    
    def batch_predict(self, texts: list) -> list:
        """Batch prediction"""
        print(f"[NeMoModel] Received batch of {len(texts)} texts")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.batch_predict(texts)
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            print(f"[NeMoModel] Returning batch results: {results}")
            return results
            
        except Exception as e:
            print(f"[NeMoModel] Error during batch prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.batch_predict(texts)


class NeMoQuartzNetModel(BaseModel):
    """QuartzNet15x5 model for speech recognition and sentiment analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._model = None
        self._sentiment_model = None
        print(f"[NeMoQuartzNetModel] Initialized with config: {config}")
    
    def _load_model(self):
        """Lazy load the QuartzNet model"""
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
                from nemo.collections.asr.models import EncDecCTCModel
                
                model_name = self.model_config.get("model_name", "nvidia/quartznet15x5")
                print(f"[NeMoQuartzNetModel] Loading QuartzNet model: {model_name}")
                
                # Load QuartzNet model
                self._model = EncDecCTCModel.from_pretrained(model_name)
                
                # Set device
                device = self.model_config.get("device", "auto")
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._model = self._model.to(device)
                self._model.eval()
                
                print(f"[NeMoQuartzNetModel] QuartzNet model loaded successfully on {device}")
                
                # Initialize sentiment model for text analysis
                self._sentiment_model = DummyModel(self.config)
                
            except ImportError:
                print("[NeMoQuartzNetModel] Warning: nemo not installed, falling back to dummy model")
                return DummyModel(self.config)
            except Exception as e:
                print(f"[NeMoQuartzNetModel] Error loading model: {e}, falling back to dummy model")
                return DummyModel(self.config)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using QuartzNet (simulated for text input)"""
        print(f"[NeMoQuartzNetModel] Received text: '{text}'")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.predict(text)
        
        try:
            # For text input, we simulate the speech-to-text process
            # In a real scenario, this would process audio files
            print(f"[NeMoQuartzNetModel] Simulating speech-to-text for: '{text}'")
            
            # Use sentiment model for text analysis
            sentiment_result = self._sentiment_model.predict(text)
            
            # Add QuartzNet-specific metadata
            output = {
                **sentiment_result,
                "model_type": "quartznet15x5",
                "speech_to_text": text,  # In real scenario, this would be transcribed text
                "audio_processed": False,  # Flag indicating this was text input
                "sample_rate": self.model_config.get("sample_rate", 16000),
                "max_audio_length": self.model_config.get("max_audio_length", 30)
            }
            
            print(f"[NeMoQuartzNetModel] Returning: {output}")
            return output
            
        except Exception as e:
            print(f"[NeMoQuartzNetModel] Error during prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.predict(text)
    
    def batch_predict(self, texts: list) -> list:
        """Batch prediction"""
        print(f"[NeMoQuartzNetModel] Received batch of {len(texts)} texts")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.batch_predict(texts)
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            print(f"[NeMoQuartzNetModel] Returning batch results: {results}")
            return results
            
        except Exception as e:
            print(f"[NeMoQuartzNetModel] Error during batch prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.batch_predict(texts)


class NeMoConformerModel(BaseModel):
    """ConformerCTC model for speech recognition and sentiment analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get("config", {})
        self._model = None
        self._sentiment_model = None
        print(f"[NeMoConformerModel] Initialized with config: {config}")
    
    def _load_model(self):
        """Lazy load the ConformerCTC model"""
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
                from nemo.collections.asr.models import EncDecCTCModel
                
                model_name = self.model_config.get("model_name", "nvidia/stt_en_conformer_ctc_large")
                print(f"[NeMoConformerModel] Loading ConformerCTC model: {model_name}")
                
                # Load ConformerCTC model
                self._model = EncDecCTCModel.from_pretrained(model_name)
                
                # Set device
                device = self.model_config.get("device", "auto")
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._model = self._model.to(device)
                self._model.eval()
                
                print(f"[NeMoConformerModel] ConformerCTC model loaded successfully on {device}")
                
                # Initialize sentiment model for text analysis
                self._sentiment_model = DummyModel(self.config)
                
            except ImportError:
                print("[NeMoConformerModel] Warning: nemo not installed, falling back to dummy model")
                return DummyModel(self.config)
            except Exception as e:
                print(f"[NeMoConformerModel] Error loading model: {e}, falling back to dummy model")
                return DummyModel(self.config)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using ConformerCTC (simulated for text input)"""
        print(f"[NeMoConformerModel] Received text: '{text}'")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.predict(text)
        
        try:
            # For text input, we simulate the speech-to-text process
            # In a real scenario, this would process audio files
            print(f"[NeMoConformerModel] Simulating speech-to-text for: '{text}'")
            
            # Use sentiment model for text analysis
            sentiment_result = self._sentiment_model.predict(text)
            
            # Add ConformerCTC-specific metadata
            output = {
                **sentiment_result,
                "model_type": "conformer_ctc",
                "speech_to_text": text,  # In real scenario, this would be transcribed text
                "audio_processed": False,  # Flag indicating this was text input
                "sample_rate": self.model_config.get("sample_rate", 16000),
                "max_audio_length": self.model_config.get("max_audio_length", 30)
            }
            
            print(f"[NeMoConformerModel] Returning: {output}")
            return output
            
        except Exception as e:
            print(f"[NeMoConformerModel] Error during prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.predict(text)
    
    def batch_predict(self, texts: list) -> list:
        """Batch prediction"""
        print(f"[NeMoConformerModel] Received batch of {len(texts)} texts")
        
        # Load model if not already loaded
        if self._model is None:
            self._load_model()
            if isinstance(self._model, DummyModel):
                return self._model.batch_predict(texts)
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            print(f"[NeMoConformerModel] Returning batch results: {results}")
            return results
            
        except Exception as e:
            print(f"[NeMoConformerModel] Error during batch prediction: {e}, falling back to dummy")
            dummy_model = DummyModel(self.config)
            return dummy_model.batch_predict(texts)


class ModelFactory:
    """Factory class for creating model instances based on configuration"""
    
    def __init__(self, config_path: str = "config/models.yaml", use_wandb: bool = False):
        self.config_path = config_path
        self.config = self._load_config()
        self.use_wandb = use_wandb
        self.wandb_manager = None
        
        if use_wandb:
            try:
                from utils.wandb_utils import WandbArtifactManager
                self.wandb_manager = WandbArtifactManager()
                print("[ModelFactory] W&B integration enabled")
            except ImportError:
                print("[ModelFactory] Warning: W&B not available, continuing without artifact tracking")
        
        print(f"[ModelFactory] Initialized with config from: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[ModelFactory] Configuration loaded successfully")
            return config
        except FileNotFoundError:
            print(f"[ModelFactory] Warning: Config file {self.config_path} not found, using default config")
            return {"models": {"dummy": {"type": "dummy"}}, "default_model": "dummy"}
        except Exception as e:
            print(f"[ModelFactory] Error loading config: {e}, using default config")
            return {"models": {"dummy": {"type": "dummy"}}, "default_model": "dummy"}
    
    def get_model(self, model_name: Optional[str] = None) -> BaseModel:
        """
        Get a model instance by name
        
        Args:
            model_name (str, optional): Name of the model to load. If None, uses default.
            
        Returns:
            BaseModel: Model instance
        """
        if model_name is None:
            model_name = self.config.get("default_model", "dummy")
        
        print(f"[ModelFactory] Loading model: {model_name}")
        
        if model_name not in self.config.get("models", {}):
            print(f"[ModelFactory] Warning: Model '{model_name}' not found in config, using dummy")
            model_name = "dummy"
        
        model_config = self.config["models"][model_name]
        model_type = model_config.get("type", "dummy")
        
        # Log model artifact if W&B is enabled
        if self.use_wandb and self.wandb_manager:
            try:
                model_artifact = self.wandb_manager.log_model_artifact(
                    f"models/{model_name}_model.json",
                    model_name,
                    model_type,
                    {"config": model_config}
                )
                if model_artifact:
                    print(f"[ModelFactory] Logged model artifact: {model_artifact}")
            except Exception as e:
                print(f"[ModelFactory] Warning: Failed to log model artifact: {e}")
        
        # Create model based on type
        if model_type == "dummy":
            return DummyModel(model_config)
        elif model_type == "huggingface":
            return HuggingFaceModel(model_config)
        elif model_type == "nemo":
            return NeMoModel(model_config)
        elif model_type == "nemo_quartznet":
            return NeMoQuartzNetModel(model_config)
        elif model_type == "nemo_conformer":
            return NeMoConformerModel(model_config)
        elif model_type == "video_sentiment":
            from models.video_sentiment_model import VideoSentimentModel
            return VideoSentimentModel(model_config)
        elif model_type == "audio_emotion":
            from models.audio_emotion_model import AudioEmotionModel
            return AudioEmotionModel(model_config)
        else:
            print(f"[ModelFactory] Warning: Unknown model type '{model_type}', using dummy")
            return DummyModel(model_config)
    
    def list_available_models(self) -> list:
        """List all available models in configuration"""
        return list(self.config.get("models", {}).keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name in self.config.get("models", {}):
            return self.config["models"][model_name]
        return {}
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        print("[ModelFactory] Configuration reloaded") 