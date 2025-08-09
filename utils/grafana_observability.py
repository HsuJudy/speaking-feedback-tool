"""
Grafana Observability Integration
Sends metrics to Grafana for monitoring ML pipeline performance
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np
from dataclasses import asdict
import threading
from queue import Queue
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrafanaObservability:
    """Grafana observability integration for ML pipeline monitoring"""
    
    def __init__(self, 
                 grafana_url: str = "http://localhost:3000",
                 api_key: Optional[str] = None,
                 org_id: int = 1,
                 enabled: bool = True):
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key or os.getenv('GRAFANA_API_KEY')
        self.org_id = org_id
        self.enabled = enabled
        
        # Metrics queue for batch processing
        self.metrics_queue = Queue()
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        
        # Start background processor
        if self.enabled:
            self._start_background_processor()
        
        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
    
    def _start_background_processor(self):
        """Start background thread for processing metrics"""
        self.processor_thread = threading.Thread(target=self._process_metrics_queue, daemon=True)
        self.processor_thread.start()
        logger.info("Started Grafana metrics processor")
    
    def _process_metrics_queue(self):
        """Process metrics from queue in batches"""
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Get metric from queue with timeout
                metric = self.metrics_queue.get(timeout=1.0)
                batch.append(metric)
                
                # Send batch if full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_batch_time >= self.batch_timeout)):
                    
                    self._send_metrics_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error processing metrics queue: {e}")
                time.sleep(1)
    
    def _send_metrics_batch(self, metrics_batch: List[Dict]):
        """Send batch of metrics to Grafana"""
        if not self.enabled or not metrics_batch:
            return
        
        try:
            # Prepare metrics for Grafana
            grafana_metrics = []
            
            for metric in metrics_batch:
                grafana_metric = {
                    "time": metric.get("timestamp", int(time.time() * 1000)),
                    "value": metric.get("value", 0),
                    "metric": metric.get("metric_name", "unknown"),
                    "tags": metric.get("tags", {})
                }
                grafana_metrics.append(grafana_metric)
            
            # Send to Grafana API
            self._send_to_grafana(grafana_metrics)
            
        except Exception as e:
            logger.error(f"Error sending metrics batch: {e}")
    
    def _send_to_grafana(self, metrics: List[Dict]):
        """Send metrics to Grafana API"""
        if not self.enabled:
            return
        
        try:
            # This is a simplified implementation
            # In a real scenario, you'd use Grafana's metrics API or a time-series database
            url = f"{self.grafana_url}/api/datasources/proxy/1/api/v1/write"
            
            # Convert to InfluxDB line protocol format
            lines = []
            for metric in metrics:
                tags = ','.join([f"{k}={v}" for k, v in metric.get("tags", {}).items()])
                line = f"{metric['metric']},{tags} value={metric['value']} {metric['time']}"
                lines.append(line)
            
            payload = '\n'.join(lines)
            
            response = requests.post(
                url,
                data=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.debug(f"Sent {len(metrics)} metrics to Grafana")
            else:
                logger.warning(f"Failed to send metrics: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending to Grafana: {e}")
    
    def log_model_metrics(self, 
                         model_name: str,
                         accuracy: float,
                         precision: float,
                         recall: float,
                         f1_score: float,
                         training_time: float,
                         inference_time: float,
                         feature_importance: Dict[str, float],
                         confidence_scores: Optional[List[float]] = None,
                         shap_values: Optional[np.ndarray] = None,
                         pipeline_timing: Optional[Dict[str, float]] = None):
        """
        Log model performance metrics
        
        Args:
            model_name (str): Name of the model
            accuracy (float): Model accuracy
            precision (float): Model precision
            recall (float): Model recall
            f1_score (float): Model F1 score
            training_time (float): Training time in seconds
            inference_time (float): Inference time in seconds
            feature_importance (Dict): Feature importance scores
            confidence_scores (List): Confidence scores for predictions
            shap_values (np.ndarray): SHAP values for explainability
            pipeline_timing (Dict): Pipeline timing breakdown
        """
        timestamp = int(time.time() * 1000)
        
        # Basic model metrics
        metrics = [
            {
                "metric_name": "model_accuracy",
                "value": accuracy,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "performance"}
            },
            {
                "metric_name": "model_precision",
                "value": precision,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "performance"}
            },
            {
                "metric_name": "model_recall",
                "value": recall,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "performance"}
            },
            {
                "metric_name": "model_f1_score",
                "value": f1_score,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "performance"}
            },
            {
                "metric_name": "model_training_time",
                "value": training_time,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "timing"}
            },
            {
                "metric_name": "model_inference_time",
                "value": inference_time,
                "timestamp": timestamp,
                "tags": {"model": model_name, "metric_type": "timing"}
            }
        ]
        
        # Feature importance metrics
        for feature, importance in feature_importance.items():
            metrics.append({
                "metric_name": "feature_importance",
                "value": importance,
                "timestamp": timestamp,
                "tags": {"model": model_name, "feature": feature, "metric_type": "feature"}
            })
        
        # Confidence scores
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            std_confidence = np.std(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)
            
            metrics.extend([
                {
                    "metric_name": "confidence_mean",
                    "value": avg_confidence,
                    "timestamp": timestamp,
                    "tags": {"model": model_name, "metric_type": "confidence"}
                },
                {
                    "metric_name": "confidence_std",
                    "value": std_confidence,
                    "timestamp": timestamp,
                    "tags": {"model": model_name, "metric_type": "confidence"}
                },
                {
                    "metric_name": "confidence_min",
                    "value": min_confidence,
                    "timestamp": timestamp,
                    "tags": {"model": model_name, "metric_type": "confidence"}
                },
                {
                    "metric_name": "confidence_max",
                    "value": max_confidence,
                    "timestamp": timestamp,
                    "tags": {"model": model_name, "metric_type": "confidence"}
                }
            ])
        
        # SHAP values (simplified - just mean absolute values)
        if shap_values is not None:
            if len(shap_values.shape) == 2:
                # For 2D SHAP values, take mean across samples
                shap_mean = np.mean(np.abs(shap_values), axis=0)
                for i, value in enumerate(shap_mean):
                    metrics.append({
                        "metric_name": "shap_importance",
                        "value": float(value),
                        "timestamp": timestamp,
                        "tags": {"model": model_name, "feature_idx": i, "metric_type": "shap"}
                    })
        
        # Pipeline timing
        if pipeline_timing:
            for stage, timing in pipeline_timing.items():
                metrics.append({
                    "metric_name": "pipeline_timing",
                    "value": timing,
                    "timestamp": timestamp,
                    "tags": {"model": model_name, "stage": stage, "metric_type": "pipeline"}
                })
        
        # Add metrics to queue
        for metric in metrics:
            self.metrics_queue.put(metric)
        
        logger.info(f"Logged {len(metrics)} metrics for {model_name}")
    
    def log_prediction(self, 
                      model_name: str,
                      emotion: str,
                      confidence: float,
                      probabilities: Dict[str, float],
                      inference_time: float,
                      features: Dict[str, float]):
        """
        Log individual prediction metrics
        
        Args:
            model_name (str): Name of the model
            emotion (str): Predicted emotion
            confidence (float): Prediction confidence
            probabilities (Dict): All emotion probabilities
            inference_time (float): Inference time
            features (Dict): Input features
        """
        timestamp = int(time.time() * 1000)
        
        metrics = [
            {
                "metric_name": "prediction_confidence",
                "value": confidence,
                "timestamp": timestamp,
                "tags": {"model": model_name, "emotion": emotion, "metric_type": "prediction"}
            },
            {
                "metric_name": "prediction_inference_time",
                "value": inference_time,
                "timestamp": timestamp,
                "tags": {"model": model_name, "emotion": emotion, "metric_type": "prediction"}
            }
        ]
        
        # Log probabilities for each emotion
        for emotion_name, prob in probabilities.items():
            metrics.append({
                "metric_name": "emotion_probability",
                "value": prob,
                "timestamp": timestamp,
                "tags": {"model": model_name, "emotion": emotion_name, "metric_type": "probability"}
            })
        
        # Log feature values
        for feature_name, value in features.items():
            metrics.append({
                "metric_name": "input_feature",
                "value": value,
                "timestamp": timestamp,
                "tags": {"model": model_name, "feature": feature_name, "metric_type": "input"}
            })
        
        # Add metrics to queue
        for metric in metrics:
            self.metrics_queue.put(metric)
        
        logger.debug(f"Logged prediction metrics for {model_name} -> {emotion}")
    
    def log_pipeline_timing(self, 
                           pipeline_name: str,
                           total_time: float,
                           stage_timings: Dict[str, float]):
        """
        Log pipeline timing metrics
        
        Args:
            pipeline_name (str): Name of the pipeline
            total_time (float): Total pipeline time
            stage_timings (Dict): Timing for each pipeline stage
        """
        timestamp = int(time.time() * 1000)
        
        metrics = [
            {
                "metric_name": "pipeline_total_time",
                "value": total_time,
                "timestamp": timestamp,
                "tags": {"pipeline": pipeline_name, "metric_type": "pipeline"}
            }
        ]
        
        # Individual stage timings
        for stage, timing in stage_timings.items():
            metrics.append({
                "metric_name": "pipeline_stage_time",
                "value": timing,
                "timestamp": timestamp,
                "tags": {"pipeline": pipeline_name, "stage": stage, "metric_type": "pipeline"}
            })
        
        # Add metrics to queue
        for metric in metrics:
            self.metrics_queue.put(metric)
        
        logger.info(f"Logged pipeline timing for {pipeline_name}: {total_time:.3f}s")
    
    def create_grafana_dashboard(self) -> Dict[str, Any]:
        """
        Create Grafana dashboard configuration for sentiment observability
        
        Returns:
            Dict: Dashboard configuration
        """
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Team Sentiment Observability",
                "tags": ["sentiment", "team", "observability", "burnout"],
                "timezone": "browser",
                "panels": [
                    # Team Emotion Distribution Panel
                    {
                        "id": 1,
                        "title": "Team Emotion Distribution",
                        "type": "piechart",
                        "targets": [
                            {
                                "expr": "emotion_distribution",
                                "legendFormat": "{{emotion}} - {{count}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "mappings": [],
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    # Burnout Risk Trend Panel
                    {
                        "id": 2,
                        "title": "Burnout Risk Trend",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "burnout_risk_score",
                                "legendFormat": "{{speaker_id}} - {{team_id}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "mappings": [],
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 0.5},
                                        {"color": "red", "value": 0.7}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    # Sentiment Drift Panel
                    {
                        "id": 3,
                        "title": "Sentiment Drift Analysis",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "sentiment_drift_score",
                                "legendFormat": "{{speaker_id}} - {{team_id}}"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "mappings": [],
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": -0.3},
                                        {"color": "green", "value": 0.3}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    # Speaker-Level Breakdown Panel
                    {
                        "id": 4,
                        "title": "Speaker-Level Sentiment",
                        "type": "table",
                        "targets": [
                            {
                                "expr": "speaker_sentiment_breakdown",
                                "format": "table"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "mappings": [],
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    # Feature Importance Panel
                    {
                        "id": 2,
                        "title": "Feature Importance",
                        "type": "barchart",
                        "targets": [
                            {
                                "expr": "feature_importance",
                                "legendFormat": "{{feature}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    # Confidence Distribution Panel
                    {
                        "id": 3,
                        "title": "Prediction Confidence",
                        "type": "histogram",
                        "targets": [
                            {
                                "expr": "prediction_confidence",
                                "legendFormat": "{{model}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    # Pipeline Timing Panel
                    {
                        "id": 4,
                        "title": "Pipeline Timing",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "pipeline_stage_time",
                                "legendFormat": "{{pipeline}} - {{stage}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            },
            "folderId": 0,
            "overwrite": True
        }
        
        return dashboard
    
    def setup_grafana_dashboard(self) -> bool:
        """
        Set up Grafana dashboard
        
        Returns:
            bool: True if successful
        """
        if not self.enabled:
            logger.warning("Grafana observability is disabled")
            return False
        
        try:
            dashboard_config = self.create_grafana_dashboard()
            
            url = f"{self.grafana_url}/api/dashboards/db"
            
            response = requests.post(
                url,
                json=dashboard_config,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Dashboard created: {result.get('url', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to create dashboard: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up Grafana dashboard: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check Grafana connectivity
        
        Returns:
            bool: True if healthy
        """
        if not self.enabled:
            return True
        
        try:
            url = f"{self.grafana_url}/api/health"
            response = requests.get(url, headers=self.headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Grafana health check failed: {e}")
            return False


def test_grafana_observability():
    """Test Grafana observability integration"""
    print("📊 TESTING GRAFANA OBSERVABILITY")
    print("=" * 40)
    
    # Initialize with disabled mode for testing
    grafana = GrafanaObservability(enabled=False)
    
    # Test model metrics logging
    feature_importance = {
        "duration": 0.15,
        "rms_energy": 0.25,
        "spectral_centroid": 0.30,
        "mfcc_1": 0.20,
        "mfcc_2": 0.10
    }
    
    confidence_scores = [0.85, 0.92, 0.78, 0.95, 0.88]
    
    grafana.log_model_metrics(
        model_name="test_model",
        accuracy=0.85,
        precision=0.87,
        recall=0.83,
        f1_score=0.85,
        training_time=120.5,
        inference_time=0.002,
        feature_importance=feature_importance,
        confidence_scores=confidence_scores
    )
    
    # Test prediction logging
    grafana.log_prediction(
        model_name="test_model",
        emotion="happy",
        confidence=0.92,
        probabilities={"happy": 0.92, "sad": 0.05, "neutral": 0.03},
        inference_time=0.001,
        features={"duration": 3.5, "rms_energy": 0.15}
    )
    
    # Test pipeline timing
    grafana.log_pipeline_timing(
        pipeline_name="emotion_pipeline",
        total_time=2.5,
        stage_timings={
            "audio_loading": 0.5,
            "feature_extraction": 1.2,
            "model_inference": 0.8
        }
    )
    
    print("✅ Grafana observability test completed!")


if __name__ == "__main__":
    test_grafana_observability() 