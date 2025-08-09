"""
Real-time Sentiment Logging System
Logs each emotion inference with comprehensive metadata for observability
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import os
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentLogEntry:
    """Data class for sentiment log entries"""
    timestamp: float
    speaker_id: str
    emotion_prediction: str
    confidence_score: float
    shap_explanation: Dict[str, float]
    audio_features: Dict[str, float]
    inference_time: float
    model_name: str
    session_id: str
    team_id: str = "default"
    burnout_risk_score: Optional[float] = None
    sentiment_drift_score: Optional[float] = None


class SentimentLogger:
    """Real-time sentiment logging system for team calls and voice clips"""
    
    def __init__(self, 
                 db_path: str = "sentiment_logs.db",
                 enable_shap: bool = True,
                 enable_burnout_detection: bool = True,
                 batch_size: int = 10,
                 batch_timeout: float = 5.0):
        self.db_path = db_path
        self.enable_shap = enable_shap
        self.enable_burnout_detection = enable_burnout_detection
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Initialize database
        self._init_database()
        
        # Metrics queue for batch processing
        self.logs_queue = Queue()
        
        # Start background processor
        self._start_background_processor()
        
        logger.info("SentimentLogger initialized")
    
    def _init_database(self):
        """Initialize SQLite database for sentiment logs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sentiment_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    speaker_id TEXT NOT NULL,
                    emotion_prediction TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    shap_explanation TEXT,
                    audio_features TEXT,
                    inference_time REAL,
                    model_name TEXT,
                    session_id TEXT,
                    team_id TEXT DEFAULT 'default',
                    burnout_risk_score REAL,
                    sentiment_drift_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON sentiment_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_speaker_id ON sentiment_logs(speaker_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_id ON sentiment_logs(team_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion ON sentiment_logs(emotion_prediction)')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _start_background_processor(self):
        """Start background thread for processing logs"""
        self.processor_thread = threading.Thread(target=self._process_logs_queue, daemon=True)
        self.processor_thread.start()
        logger.info("Started sentiment logs processor")
    
    def _process_logs_queue(self):
        """Process logs from queue in batches"""
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Get log from queue with timeout
                log_entry = self.logs_queue.get(timeout=1.0)
                batch.append(log_entry)
                
                # Process batch if full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_batch_time >= self.batch_timeout)):
                    
                    self._save_logs_batch(batch)
                    batch = []
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Error processing logs queue: {e}")
                time.sleep(1)
    
    def _save_logs_batch(self, logs_batch: List[SentimentLogEntry]):
        """Save batch of logs to database"""
        if not logs_batch:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for log_entry in logs_batch:
                cursor.execute('''
                    INSERT INTO sentiment_logs (
                        timestamp, speaker_id, emotion_prediction, confidence_score,
                        shap_explanation, audio_features, inference_time, model_name,
                        session_id, team_id, burnout_risk_score, sentiment_drift_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_entry.timestamp,
                    log_entry.speaker_id,
                    log_entry.emotion_prediction,
                    log_entry.confidence_score,
                    json.dumps(log_entry.shap_explanation),
                    json.dumps(log_entry.audio_features),
                    log_entry.inference_time,
                    log_entry.model_name,
                    log_entry.session_id,
                    log_entry.team_id,
                    log_entry.burnout_risk_score,
                    log_entry.sentiment_drift_score
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(logs_batch)} sentiment logs to database")
            
        except Exception as e:
            logger.error(f"Error saving logs batch: {e}")
    
    def log_sentiment_inference(self,
                               speaker_id: str,
                               emotion_prediction: str,
                               confidence_score: float,
                               audio_features: Dict[str, float],
                               inference_time: float,
                               model_name: str,
                               session_id: Optional[str] = None,
                               team_id: str = "default",
                               shap_explanation: Optional[Dict[str, float]] = None) -> str:
        """
        Log a sentiment inference with comprehensive metadata
        
        Args:
            speaker_id: Identifier for the speaker
            emotion_prediction: Predicted emotion (calm, anxious, frustrated, energetic, burned_out)
            confidence_score: Model confidence (0-1)
            audio_features: Extracted audio features
            inference_time: Time taken for inference
            model_name: Name of the model used
            session_id: Optional session identifier
            team_id: Team identifier
            shap_explanation: Optional SHAP feature importance
            
        Returns:
            str: Log entry ID
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Calculate burnout risk score
        burnout_risk_score = self._calculate_burnout_risk(
            emotion_prediction, audio_features, confidence_score
        ) if self.enable_burnout_detection else None
        
        # Calculate sentiment drift score
        sentiment_drift_score = self._calculate_sentiment_drift(
            speaker_id, emotion_prediction, confidence_score
        )
        
        # Create log entry
        log_entry = SentimentLogEntry(
            timestamp=time.time(),
            speaker_id=speaker_id,
            emotion_prediction=emotion_prediction,
            confidence_score=confidence_score,
            shap_explanation=shap_explanation or {},
            audio_features=audio_features,
            inference_time=inference_time,
            model_name=model_name,
            session_id=session_id,
            team_id=team_id,
            burnout_risk_score=burnout_risk_score,
            sentiment_drift_score=sentiment_drift_score
        )
        
        # Add to processing queue
        self.logs_queue.put(log_entry)
        
        logger.info(f"Logged sentiment inference: {speaker_id} -> {emotion_prediction} ({confidence_score:.3f})")
        return session_id
    
    def _calculate_burnout_risk(self, 
                               emotion: str, 
                               audio_features: Dict[str, float], 
                               confidence: float) -> float:
        """
        Calculate burnout risk score based on emotion and audio features
        
        Returns:
            float: Burnout risk score (0-1, higher = higher risk)
        """
        # Base risk from emotion
        emotion_risk = {
            "burned_out": 0.9,
            "frustrated": 0.7,
            "anxious": 0.6,
            "calm": 0.2,
            "energetic": 0.1
        }.get(emotion, 0.5)
        
        # Audio feature contributions
        audio_risk = 0.0
        
        # Low energy indicators
        if audio_features.get("volume_mean", 0) < 0.3:
            audio_risk += 0.2
        
        # Monotone speaking (low pitch variability)
        if audio_features.get("pitch_std", 0) < 20:
            audio_risk += 0.15
        
        # Slow speech rate
        if audio_features.get("speech_rate", 0) < 0.4:
            audio_risk += 0.1
        
        # High silence ratio
        if audio_features.get("silence_ratio", 0) > 0.5:
            audio_risk += 0.15
        
        # Combine with confidence
        final_risk = (emotion_risk * 0.6 + audio_risk * 0.4) * confidence
        return min(final_risk, 1.0)
    
    def _calculate_sentiment_drift(self, 
                                  speaker_id: str, 
                                  emotion: str, 
                                  confidence: float) -> float:
        """
        Calculate sentiment drift score for a speaker
        
        Returns:
            float: Sentiment drift score (-1 to 1, negative = pessimistic drift)
        """
        # Get recent history for this speaker
        recent_emotions = self._get_recent_emotions(speaker_id, limit=10)
        
        if not recent_emotions:
            return 0.0
        
        # Calculate drift based on emotion transitions
        emotion_scores = {
            "energetic": 1.0,
            "calm": 0.5,
            "anxious": -0.3,
            "frustrated": -0.7,
            "burned_out": -1.0
        }
        
        current_score = emotion_scores.get(emotion, 0.0) * confidence
        avg_historical_score = sum(emotion_scores.get(e, 0.0) for e in recent_emotions) / len(recent_emotions)
        
        drift = current_score - avg_historical_score
        return max(-1.0, min(1.0, drift))
    
    def _get_recent_emotions(self, speaker_id: str, limit: int = 10) -> List[str]:
        """Get recent emotions for a speaker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT emotion_prediction 
                FROM sentiment_logs 
                WHERE speaker_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (speaker_id, limit))
            
            emotions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return emotions
            
        except Exception as e:
            logger.error(f"Error getting recent emotions: {e}")
            return []
    
    def get_team_sentiment_summary(self, team_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get sentiment summary for a team over specified hours
        
        Args:
            team_id: Team identifier
            hours: Number of hours to analyze
            
        Returns:
            Dict containing team sentiment summary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get data from last N hours
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT emotion_prediction, confidence_score, burnout_risk_score, sentiment_drift_score
                FROM sentiment_logs 
                WHERE team_id = ? AND timestamp > ?
            ''', (team_id, cutoff_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"error": "No data found for team"}
            
            # Calculate statistics
            emotions = [row[0] for row in rows]
            confidences = [row[1] for row in rows]
            burnout_scores = [row[2] for row in rows if row[2] is not None]
            drift_scores = [row[3] for row in rows if row[3] is not None]
            
            # Emotion distribution
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Calculate averages
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_burnout_risk = sum(burnout_scores) / len(burnout_scores) if burnout_scores else 0
            avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 0
            
            return {
                "team_id": team_id,
                "time_period_hours": hours,
                "total_inferences": len(rows),
                "emotion_distribution": emotion_counts,
                "avg_confidence": avg_confidence,
                "avg_burnout_risk": avg_burnout_risk,
                "avg_sentiment_drift": avg_drift,
                "burnout_alert": avg_burnout_risk > 0.7,
                "drift_alert": abs(avg_drift) > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting team sentiment summary: {e}")
            return {"error": str(e)}
    
    def get_speaker_breakdown(self, team_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get speaker-level sentiment breakdown
        
        Args:
            team_id: Team identifier
            hours: Number of hours to analyze
            
        Returns:
            Dict containing speaker-level breakdown
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT speaker_id, emotion_prediction, confidence_score, burnout_risk_score
                FROM sentiment_logs 
                WHERE team_id = ? AND timestamp > ?
            ''', (team_id, cutoff_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"error": "No data found for team"}
            
            # Group by speaker
            speaker_data = {}
            for row in rows:
                speaker_id, emotion, confidence, burnout_risk = row
                
                if speaker_id not in speaker_data:
                    speaker_data[speaker_id] = {
                        "emotions": [],
                        "confidences": [],
                        "burnout_risks": []
                    }
                
                speaker_data[speaker_id]["emotions"].append(emotion)
                speaker_data[speaker_id]["confidences"].append(confidence)
                if burnout_risk is not None:
                    speaker_data[speaker_id]["burnout_risks"].append(burnout_risk)
            
            # Calculate speaker statistics
            speaker_summary = {}
            for speaker_id, data in speaker_data.items():
                emotions = data["emotions"]
                confidences = data["confidences"]
                burnout_risks = data["burnout_risks"]
                
                # Emotion distribution
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Most common emotion
                most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "unknown"
                
                # Averages
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_burnout_risk = sum(burnout_risks) / len(burnout_risks) if burnout_risks else 0
                
                speaker_summary[speaker_id] = {
                    "total_inferences": len(emotions),
                    "most_common_emotion": most_common_emotion,
                    "emotion_distribution": emotion_counts,
                    "avg_confidence": avg_confidence,
                    "avg_burnout_risk": avg_burnout_risk,
                    "burnout_alert": avg_burnout_risk > 0.7
                }
            
            return {
                "team_id": team_id,
                "time_period_hours": hours,
                "speaker_breakdown": speaker_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting speaker breakdown: {e}")
            return {"error": str(e)}


def test_sentiment_logger():
    """Test the sentiment logger"""
    print("📊 TESTING SENTIMENT LOGGER")
    print("=" * 40)
    
    # Initialize logger
    logger = SentimentLogger(db_path=":memory:")  # Use in-memory DB for testing
    
    # Test logging
    print("Logging sentiment inferences...")
    
    # Mock audio features
    audio_features = {
        "pitch_mean": 200.0,
        "volume_mean": 0.5,
        "speech_rate": 0.7,
        "silence_ratio": 0.3
    }
    
    # Mock SHAP explanation
    shap_explanation = {
        "pitch_mean": 0.3,
        "volume_mean": 0.2,
        "speech_rate": 0.4,
        "silence_ratio": 0.1
    }
    
    # Log some test inferences
    session_id = logger.log_sentiment_inference(
        speaker_id="speaker_001",
        emotion_prediction="energetic",
        confidence_score=0.85,
        audio_features=audio_features,
        inference_time=0.002,
        model_name="test_model",
        team_id="team_a",
        shap_explanation=shap_explanation
    )
    
    logger.log_sentiment_inference(
        speaker_id="speaker_002",
        emotion_prediction="frustrated",
        confidence_score=0.72,
        audio_features=audio_features,
        inference_time=0.001,
        model_name="test_model",
        session_id=session_id,
        team_id="team_a"
    )
    
    logger.log_sentiment_inference(
        speaker_id="speaker_001",
        emotion_prediction="burned_out",
        confidence_score=0.91,
        audio_features=audio_features,
        inference_time=0.003,
        model_name="test_model",
        session_id=session_id,
        team_id="team_a"
    )
    
    # Wait for processing
    time.sleep(2)
    
    # Test team summary
    print("\nGetting team sentiment summary...")
    summary = logger.get_team_sentiment_summary("team_a", hours=1)
    print(f"Team Summary: {summary}")
    
    # Test speaker breakdown
    print("\nGetting speaker breakdown...")
    breakdown = logger.get_speaker_breakdown("team_a", hours=1)
    print(f"Speaker Breakdown: {breakdown}")
    
    print("✅ Sentiment logger test completed!")


if __name__ == "__main__":
    test_sentiment_logger() 