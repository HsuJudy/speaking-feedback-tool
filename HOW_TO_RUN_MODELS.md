# How to Run Models in Speaking Feedback Tool

## 🚀 Quick Start - Running Models

### 1. **Run the Main Application** (Easiest)
```bash
cd vibe-check
python3 main.py
```
This runs the complete pipeline with multiple models and shows results.

### 2. **Run Individual Demos**
```bash
# Run custom models demo
python3 demo_custom_models.py

# Run inference pipeline
python3 inference.py

# Run training pipeline
python3 train.py
```

## 🤖 Model Running Options

### Option 1: **Main Application** (Complete Pipeline)
```bash
python3 main.py
```
**What it does:**
- Runs single text analysis with different models
- Runs batch analysis with multiple texts
- Tests multiple model types (conformer_ctc, video_sentiment, audio_emotion)
- Shows complete pipeline results

**Output example:**
```
📊 SINGLE RESULT (conformer_ctc):
{
  "input_text": "i love this product! its amazing and works perfectly.",
  "sentiment": "positive",
  "confidence": 0.854,
  "processed_at": "2025-08-01T15:30:16.148040",
  "model_info": {
    "version": "nemo_conformer_v1.0",
    "inference_timestamp": "2024-01-01T00:00:00Z"
  },
  "sentiment_emoji": "😊"
}
```

### Option 2: **Custom Models Demo** (Training & Inference)
```bash
python3 demo_custom_models.py
```
**What it does:**
- Creates dummy dataset
- Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Tests model inference
- Shows model comparison

**Output example:**
```
📊 MODEL COMPARISON:
------------------------------
logistic_regression:
  Accuracy: 0.190
  F1-Score: 0.123
  Training Time: 0.01s
  Inference Time: 0.0002s

random_forest:
  Accuracy: 0.150
  F1-Score: 0.133
  Training Time: 0.09s
  Inference Time: 0.0272s
```

### Option 3: **Inference Pipeline** (Model Serving)
```bash
python3 inference.py
```
**What it does:**
- Tests complete inference pipeline
- Creates dummy model for testing
- Shows prediction results with SHAP explanations
- Tests health checks and model info

**Output example:**
```
Prediction result: {
  "success": true,
  "emotion": "anxious",
  "confidence": 0.6,
  "probabilities": [0.6, 0.2, 0.1, 0.0, 0.1],
  "features": {
    "pitch_mean": 439.99999999999994,
    "volume_mean": 0.2543692886829376,
    ...
  },
  "shap_explanation": {...},
  "timing": {
    "feature_extraction": 0.5453388690948486,
    "prediction": 0.0005848407745361328,
    "total": 0.5466501712799072
  }
}
```

### Option 4: **Training Pipeline** (Model Training)
```bash
python3 train.py
```
**What it does:**
- Creates sample dataset
- Trains models with reproducible splits
- Saves models to registry
- Tests model loading and prediction

**Output example:**
```
INFO:__main__:Training random_forest model...
INFO:__main__:Model trained in 0.039s, CV score: 0.196 ± 0.028
INFO:__main__:Model accuracy: 0.270
INFO:__main__:Model saved: models/emotion_classification_test/random_forest_20250801_153009.pkl
INFO:__main__:Model registered with ID: 5782fb49-5045-4920-bc25-4e7a7187dd44
```

## 🔧 Running Specific Models

### 1. **Run Dummy Model**
```python
from models.model_factory import ModelFactory

# Initialize factory
factory = ModelFactory("config/models.yaml")

# Load dummy model
model = factory.get_model("dummy")

# Make prediction
result = model.predict("I love this product!")
print(result)
# Output: {'sentiment': 'positive', 'confidence': 0.854, 'text': 'I love this product!'}
```

### 2. **Run HuggingFace Model**
```python
from models.model_factory import ModelFactory

# Initialize factory
factory = ModelFactory("config/models.yaml")

# Load HuggingFace model
model = factory.get_model("huggingface")

# Make prediction
result = model.predict("This is amazing!")
print(result)
```

### 3. **Run Custom Trained Model**
```python
from train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Load trained model by ID
model_id = "5782fb49-5045-4920-bc25-4e7a7187dd44"
model, scaler, metadata = trainer.load_model(model_id)

# Make prediction (you'd need to prepare features)
# This requires feature extraction first
```

## 📊 Running with Different Input Types

### 1. **Text Input**
```python
# Single text prediction
result = model.predict("I love this product!")

# Batch text prediction
texts = [
    "I love this product!",
    "This is terrible!",
    "It's okay, nothing special."
]
results = model.batch_predict(texts)
```

### 2. **Audio Input** (for audio models)
```python
# Load audio file
audio_path = "test_audio.wav"

# Extract features
from utils.audio_features import AudioFeatureExtractor
extractor = AudioFeatureExtractor()
features = extractor.extract_features(audio_path)

# Make prediction
result = model.predict(features)
```

### 3. **Video Input** (for video models)
```python
# Load video file
video_path = "test_video.mp4"

# Extract audio from video
from utils.video_utils import VideoProcessor
processor = VideoProcessor()
audio = processor.extract_audio_from_video(video_path)

# Make prediction
result = model.predict(audio)
```

## 🎯 Running Different Model Types

### 1. **Dummy Model** (Testing)
```bash
python3 main.py
# Uses dummy model for testing pipeline functionality
```

### 2. **Custom Trained Models** (Real Models)
```bash
# First train models
python3 train.py

# Then run inference
python3 inference.py
```

### 3. **HuggingFace Models** (Advanced)
```bash
# Requires transformers library
pip install transformers torch

# Then run with HuggingFace model
python3 main.py
```

### 4. **NeMo Models** (Speech Recognition)
```bash
# Requires NeMo library
pip install nemo_toolkit

# Then run with NeMo models
python3 main.py
```

## 🔍 Model Configuration

### 1. **Check Available Models**
```python
from models.model_factory import ModelFactory

factory = ModelFactory("config/models.yaml")
available_models = factory.list_available_models()
print(available_models)
# Output: ['dummy', 'huggingface', 'nemo', 'quartznet15x5', 'conformer_ctc', 'audio_emotion']
```

### 2. **Get Model Configuration**
```python
model_info = factory.get_model_info("huggingface")
print(model_info)
# Output: {'type': 'huggingface', 'version': 'v1.0', 'confidence_threshold': 0.5, ...}
```

### 3. **Change Default Model**
Edit `config/models.yaml`:
```yaml
# Change default model
default_model: "huggingface"  # Instead of "dummy"
```

## 📈 Running with Observability

### 1. **Run with Sentiment Observability**
```bash
python3 sentiment_observability_pipeline.py
```
**What it does:**
- Processes audio through complete pipeline
- Logs sentiment inferences
- Updates mood map
- Shows burnout risk assessment

### 2. **Run with Grafana Monitoring**
```python
from utils.grafana_observability import GrafanaObservability

# Initialize monitoring
grafana = GrafanaObservability(enabled=True)

# Log model metrics
grafana.log_model_metrics(
    model_name="demo_model",
    accuracy=0.85,
    precision=0.87,
    recall=0.83,
    f1_score=0.85
)
```

## 🚀 Production Deployment

### 1. **Simple API Server**
```python
from flask import Flask, request, jsonify
from models.model_factory import ModelFactory

app = Flask(__name__)
factory = ModelFactory("config/models.yaml")
model = factory.get_model("huggingface")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    result = model.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. **Batch Processing**
```python
import pandas as pd
from models.model_factory import ModelFactory

# Load data
df = pd.read_csv('input_data.csv')

# Initialize model
factory = ModelFactory("config/models.yaml")
model = factory.get_model("huggingface")

# Process batch
results = []
for text in df['text']:
    result = model.predict(text)
    results.append(result)

# Save results
df['sentiment'] = [r['sentiment'] for r in results]
df['confidence'] = [r['confidence'] for r in results]
df.to_csv('output_data.csv', index=False)
```

## 🔧 Troubleshooting

### 1. **Model Not Found**
```bash
# Check if model exists in config
cat config/models.yaml

# Check if model files exist
ls models/custom/
```

### 2. **Dependencies Missing**
```bash
# Install required packages
pip install -r requirements.txt

# For specific models
pip install transformers torch  # For HuggingFace
pip install nemo_toolkit       # For NeMo
```

### 3. **Configuration Issues**
```bash
# Check configuration
python3 -c "from config import config; config.print_config()"
```

## 📊 Expected Outputs

### 1. **Single Prediction**
```json
{
  "sentiment": "positive",
  "confidence": 0.854,
  "text": "I love this product!",
  "model_version": "dummy_v1.0",
  "model_name": "dummy"
}
```

### 2. **Batch Prediction**
```json
[
  {
    "sentiment": "positive",
    "confidence": 0.885,
    "text": "I love this product!"
  },
  {
    "sentiment": "negative",
    "confidence": 0.704,
    "text": "This is terrible!"
  }
]
```

### 3. **Model Training Results**
```json
{
  "model_id": "5782fb49-5045-4920-bc25-4e7a7187dd44",
  "accuracy": 0.270,
  "training_time": 0.039,
  "model_path": "models/emotion_classification_test/random_forest_20250801_153009.pkl"
}
```

## 🎯 Quick Commands Summary

```bash
# Run complete pipeline
python3 main.py

# Train models
python3 train.py

# Test inference
python3 inference.py

# Run custom models demo
python3 demo_custom_models.py

# Run with observability
python3 sentiment_observability_pipeline.py

# Check configuration
python3 -c "from config import config; config.print_config()"
```

---

*This guide shows all the different ways to run models in your Speaking Feedback Tool, from simple demos to production deployment!* 