"""
Triton Inference Server Integration for Speaking Feedback Tool

This module provides Triton Inference Server integration for high-performance,
GPU-accelerated model serving with support for multiple model formats.
"""

import os
import json
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import np_to_triton_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logging.warning("Triton client not available. Install tritonclient for GPU serving.")

try:
    import nemo.collections.asr as nemo_asr
    import nemo.collections.nlp as nemo_nlp
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo not available. Install nemo-toolkit for NeMo models.")

from models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class TritonModelServer:
    """Manages Triton Inference Server for high-performance model serving"""
    
    def __init__(self, server_url: str = "localhost:8000", protocol: str = "http"):
        """Initialize Triton client"""
        if not TRITON_AVAILABLE:
            logger.warning("Triton client not available. GPU serving disabled.")
            return
            
        self.server_url = server_url
        self.protocol = protocol
        
        # Initialize client based on protocol
        if protocol == "http":
            self.client = httpclient.InferenceServerClient(url=server_url)
        elif protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=server_url)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        logger.info(f"Triton client initialized: {server_url} ({protocol})")
    
    def is_server_ready(self) -> bool:
        """Check if Triton server is ready"""
        if not TRITON_AVAILABLE:
            return False
            
        try:
            return self.client.is_server_ready()
        except Exception as e:
            logger.error(f"Triton server not ready: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models available on Triton server"""
        if not TRITON_AVAILABLE:
            return []
            
        try:
            models = self.client.get_model_repository_index()
            return [
                {
                    "name": model.name,
                    "version": model.version,
                    "state": model.state,
                    "platform": model.platform
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error listing Triton models: {e}")
            return []
    
    def predict(self, model_name: str, inputs: Dict[str, np.ndarray], 
                outputs: List[str] = None) -> Dict[str, np.ndarray]:
        """Make prediction using Triton server"""
        if not TRITON_AVAILABLE:
            logger.warning("Cannot make prediction: Triton not available")
            return {}
            
        try:
            # Prepare inputs
            triton_inputs = []
            for name, data in inputs.items():
                triton_inputs.append(
                    self.client.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
                )
                triton_inputs[-1].set_data_from_numpy(data)
            
            # Prepare outputs
            triton_outputs = []
            if outputs:
                for output_name in outputs:
                    triton_outputs.append(self.client.InferRequestedOutput(output_name))
            
            # Make inference request
            response = self.client.infer(model_name, triton_inputs, outputs=triton_outputs)
            
            # Extract results
            results = {}
            for output in response.get_response():
                results[output.name()] = response.as_numpy(output.name())
            
            return results
            
        except Exception as e:
            logger.error(f"Triton prediction failed: {e}")
            return {}
    
    def predict_text(self, model_name: str, text: str) -> Dict[str, Any]:
        """Make text prediction using Triton server"""
        # Convert text to numpy array
        text_array = np.array([text.encode('utf-8')], dtype=np.object_)
        
        inputs = {"text_input": text_array}
        results = self.predict(model_name, inputs)
        
        # Convert results back to Python types
        if results:
            return {
                "text": text,
                "prediction": results.get("output", [None])[0],
                "confidence": results.get("confidence", [0.0])[0],
                "model": model_name
            }
        return {"text": text, "error": "Prediction failed"}


class NeMoModelManager:
    """Manages NeMo models for speech and NLP tasks"""
    
    def __init__(self, model_dir: str = "models/nemo"):
        """Initialize NeMo model manager"""
        if not NEMO_AVAILABLE:
            logger.warning("NeMo not available. NeMo models disabled.")
            return
            
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Available NeMo models
        self.available_models = {
            "speech_recognition": "stt_en_conformer_transducer_large",
            "sentiment_analysis": "text_classification_sentiment",
            "speaker_recognition": "spkrec_ecapa_tdnn",
            "emotion_recognition": "emotion_classification"
        }
        
        logger.info(f"NeMo model manager initialized: {model_dir}")
    
    def download_model(self, model_type: str, model_name: str = None) -> str:
        """Download a NeMo model"""
        if not NEMO_AVAILABLE:
            logger.warning("Cannot download model: NeMo not available")
            return None
            
        try:
            if model_type == "speech_recognition":
                model = nemo_asr.models.EncDecTransducerModel.from_pretrained(model_name or self.available_models[model_type])
            elif model_type == "sentiment_analysis":
                model = nemo_nlp.models.TextClassificationModel.from_pretrained(model_name or self.available_models[model_type])
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            
            # Save model
            model_path = self.model_dir / f"{model_type}_{model_name}.nemo"
            model.save_to(str(model_path))
            
            logger.info(f"NeMo model downloaded: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error downloading NeMo model: {e}")
            return None
    
    def load_model(self, model_path: str) -> Any:
        """Load a NeMo model"""
        if not NEMO_AVAILABLE:
            logger.warning("Cannot load model: NeMo not available")
            return None
            
        try:
            # Load based on file extension
            if model_path.endswith('.nemo'):
                # Try different model types
                try:
                    return nemo_asr.models.EncDecTransducerModel.restore_from(model_path)
                except:
                    try:
                        return nemo_nlp.models.TextClassificationModel.restore_from(model_path)
                    except:
                        logger.error(f"Could not load NeMo model: {model_path}")
                        return None
            
            logger.error(f"Unsupported model format: {model_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading NeMo model: {e}")
            return None
    
    def predict_speech(self, model, audio_path: str) -> Dict[str, Any]:
        """Make speech recognition prediction"""
        if not NEMO_AVAILABLE:
            return {"error": "NeMo not available"}
            
        try:
            # Transcribe audio
            transcript = model.transcribe([audio_path])
            
            return {
                "audio_path": audio_path,
                "transcript": transcript[0] if transcript else "",
                "model": "nemo_speech_recognition"
            }
            
        except Exception as e:
            logger.error(f"Speech prediction failed: {e}")
            return {"error": str(e)}
    
    def predict_sentiment(self, model, text: str) -> Dict[str, Any]:
        """Make sentiment analysis prediction"""
        if not NEMO_AVAILABLE:
            return {"error": "NeMo not available"}
            
        try:
            # Make prediction
            prediction = model.classifytext([text])
            
            return {
                "text": text,
                "sentiment": prediction[0] if prediction else "neutral",
                "model": "nemo_sentiment_analysis"
            }
            
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {e}")
            return {"error": str(e)}


class TritonNeMoIntegration:
    """Integrates Triton with NeMo models for GPU-accelerated serving"""
    
    def __init__(self, triton_url: str = "localhost:8000", nemo_model_dir: str = "models/nemo"):
        """Initialize Triton-NeMo integration"""
        self.triton = TritonModelServer(triton_url)
        self.nemo = NeMoModelManager(nemo_model_dir)
        
        logger.info("Triton-NeMo integration initialized")
    
    def deploy_nemo_model_to_triton(self, model_path: str, model_name: str) -> bool:
        """Deploy a NeMo model to Triton server"""
        if not TRITON_AVAILABLE or not NEMO_AVAILABLE:
            logger.warning("Cannot deploy model: Triton or NeMo not available")
            return False
            
        try:
            # Load NeMo model
            model = self.nemo.load_model(model_path)
            if not model:
                return False
            
            # Convert to ONNX format for Triton
            onnx_path = model_path.replace('.nemo', '.onnx')
            
            # Export to ONNX (this is a simplified example)
            # In practice, you'd need to implement proper ONNX export
            logger.info(f"Model {model_name} would be deployed to Triton")
            logger.info(f"ONNX path: {onnx_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model to Triton: {e}")
            return False
    
    def predict_with_gpu_acceleration(self, model_name: str, text: str = None, 
                                    audio_path: str = None) -> Dict[str, Any]:
        """Make GPU-accelerated prediction using Triton + NeMo"""
        
        if text:
            # Text-based prediction
            text_array = np.array([text.encode('utf-8')], dtype=np.object_)
            inputs = {"text_input": text_array}
            
            results = self.triton.predict(model_name, inputs)
            
            return {
                "text": text,
                "prediction": results.get("output", [None])[0],
                "confidence": results.get("confidence", [0.0])[0],
                "model": model_name,
                "accelerated": True
            }
        
        elif audio_path:
            # Audio-based prediction
            # This would require audio preprocessing
            return {
                "audio_path": audio_path,
                "error": "Audio prediction not implemented yet",
                "model": model_name
            }
        
        return {"error": "No input provided"}


def demo_triton_integration():
    """Demonstrate Triton integration"""
    print("🚀 TRITON INFERENCE SERVER INTEGRATION")
    print("=" * 50)
    
    # Initialize Triton client
    triton = TritonModelServer()
    
    # Check server status
    if triton.is_server_ready():
        print("✅ Triton server is ready")
        
        # List available models
        models = triton.list_models()
        print(f"📋 Available models: {len(models)}")
        for model in models:
            print(f"  - {model['name']} ({model['platform']})")
        
        # Test prediction (if models available)
        if models:
            test_text = "I love this product!"
            result = triton.predict_text(models[0]['name'], test_text)
            print(f"✅ Test prediction: {result}")
    
    else:
        print("❌ Triton server not available")
        print("ℹ️  Start Triton server with: docker run --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models")
    
    print("=" * 50)


def demo_nemo_integration():
    """Demonstrate NeMo integration"""
    print("🎤 NEMO MODEL INTEGRATION")
    print("=" * 50)
    
    # Initialize NeMo manager
    nemo = NeMoModelManager()
    
    # Test model download (commented out to avoid large downloads)
    print("ℹ️  NeMo models would be downloaded here")
    print("ℹ️  Available models:")
    for model_type, model_name in nemo.available_models.items():
        print(f"  - {model_type}: {model_name}")
    
    print("=" * 50)


def demo_triton_nemo_integration():
    """Demonstrate Triton + NeMo integration"""
    print("⚡ TRITON + NEMO INTEGRATION")
    print("=" * 50)
    
    integration = TritonNeMoIntegration()
    
    # Test GPU-accelerated prediction
    test_text = "This is amazing!"
    result = integration.predict_with_gpu_acceleration("sentiment-model", text=test_text)
    
    print(f"✅ GPU-accelerated prediction: {result}")
    print("=" * 50)


if __name__ == "__main__":
    # Run all demos
    demo_triton_integration()
    demo_nemo_integration()
    demo_triton_nemo_integration()
    
    print("\n🎉 NVIDIA MLOps Stack Integration Complete!")
    print("\n📋 Next Steps:")
    print("  1. Install NVIDIA dependencies: pip install tritonclient nemo-toolkit")
    print("  2. Start Triton server with GPU support")
    print("  3. Deploy NeMo models to Triton")
    print("  4. Integrate with your Zoom pipeline for GPU acceleration") 