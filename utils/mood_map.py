"""
Dynamic Mood Map Prototype
Generates real-time team mood maps with emojis and heat visualization
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MoodMapCell:
    """Represents a cell in the mood map"""
    speaker_id: str
    emotion: str
    confidence: float
    burnout_risk: float
    last_updated: float
    position: Tuple[int, int]
    emoji: str
    color: str


class DynamicMoodMap:
    """Dynamic mood map for team sentiment visualization"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (5, 5),
                 update_interval: float = 5.0,
                 decay_rate: float = 0.1):
        self.grid_size = grid_size
        self.update_interval = update_interval
        self.decay_rate = decay_rate
        self.cells: Dict[str, MoodMapCell] = {}
        self.team_data: Dict[str, Dict] = {}
        self.last_update = time.time()
        
        # Emotion to emoji mapping
        self.emotion_emojis = {
            "energetic": "⚡",
            "calm": "😌",
            "anxious": "😰",
            "frustrated": "😤",
            "burned_out": "😵"
        }
        
        # Emotion to color mapping (hex colors)
        self.emotion_colors = {
            "energetic": "#FFD700",  # Gold
            "calm": "#98FB98",       # Pale green
            "anxious": "#FFB6C1",    # Light pink
            "frustrated": "#FF6347", # Tomato red
            "burned_out": "#8B0000"  # Dark red
        }
        
        logger.info(f"MoodMap initialized with grid size {grid_size}")
    
    def update_speaker_mood(self,
                           speaker_id: str,
                           emotion: str,
                           confidence: float,
                           burnout_risk: float,
                           team_id: str = "default") -> Dict[str, Any]:
        """
        Update a speaker's mood on the map
        
        Args:
            speaker_id: Speaker identifier
            emotion: Predicted emotion
            confidence: Model confidence
            burnout_risk: Burnout risk score
            team_id: Team identifier
            
        Returns:
            Dict containing updated mood map data
        """
        current_time = time.time()
        
        # Get or create position for speaker
        position = self._get_speaker_position(speaker_id)
        
        # Create or update cell
        cell = MoodMapCell(
            speaker_id=speaker_id,
            emotion=emotion,
            confidence=confidence,
            burnout_risk=burnout_risk,
            last_updated=current_time,
            position=position,
            emoji=self.emotion_emojis.get(emotion, "❓"),
            color=self.emotion_colors.get(emotion, "#808080")
        )
        
        self.cells[speaker_id] = cell
        
        # Update team data
        if team_id not in self.team_data:
            self.team_data[team_id] = {
                "speakers": {},
                "overall_mood": {},
                "burnout_alerts": [],
                "last_updated": current_time
            }
        
        self.team_data[team_id]["speakers"][speaker_id] = {
            "emotion": emotion,
            "confidence": confidence,
            "burnout_risk": burnout_risk,
            "position": position,
            "last_updated": current_time
        }
        
        # Update overall team mood
        self._update_team_mood(team_id)
        
        logger.info(f"Updated mood for {speaker_id}: {emotion} ({confidence:.3f})")
        
        return self.get_mood_map_data(team_id)
    
    def _get_speaker_position(self, speaker_id: str) -> Tuple[int, int]:
        """Get or assign position for speaker on the grid"""
        if speaker_id in self.cells:
            return self.cells[speaker_id].position
        
        # Find available position
        used_positions = {cell.position for cell in self.cells.values()}
        available_positions = []
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) not in used_positions:
                    available_positions.append((x, y))
        
        if available_positions:
            return random.choice(available_positions)
        else:
            # If grid is full, find least recently updated cell
            oldest_cell = min(self.cells.values(), key=lambda c: c.last_updated)
            return oldest_cell.position
    
    def _update_team_mood(self, team_id: str):
        """Update overall team mood statistics"""
        if team_id not in self.team_data:
            return
        
        team = self.team_data[team_id]
        speakers = team["speakers"]
        
        if not speakers:
            return
        
        # Calculate emotion distribution
        emotion_counts = {}
        total_confidence = 0
        total_burnout_risk = 0
        burnout_alerts = []
        
        for speaker_data in speakers.values():
            emotion = speaker_data["emotion"]
            confidence = speaker_data["confidence"]
            burnout_risk = speaker_data["burnout_risk"]
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += confidence
            total_burnout_risk += burnout_risk
            
            # Check for burnout alerts
            if burnout_risk > 0.7:
                burnout_alerts.append({
                    "speaker_id": speaker_data.get("speaker_id", "unknown"),
                    "burnout_risk": burnout_risk,
                    "emotion": emotion
                })
        
        # Calculate averages
        avg_confidence = total_confidence / len(speakers)
        avg_burnout_risk = total_burnout_risk / len(speakers)
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "unknown"
        
        # Update team data
        team["overall_mood"] = {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "avg_confidence": avg_confidence,
            "avg_burnout_risk": avg_burnout_risk,
            "total_speakers": len(speakers),
            "burnout_alert_count": len(burnout_alerts)
        }
        
        team["burnout_alerts"] = burnout_alerts
        team["last_updated"] = time.time()
    
    def get_mood_map_data(self, team_id: str = "default") -> Dict[str, Any]:
        """
        Get current mood map data for visualization
        
        Args:
            team_id: Team identifier
            
        Returns:
            Dict containing mood map data
        """
        current_time = time.time()
        
        # Clean up old cells
        self._cleanup_old_cells(current_time)
        
        # Prepare grid data
        grid = [[None for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        for cell in self.cells.values():
            x, y = cell.position
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                grid[x][y] = {
                    "speaker_id": cell.speaker_id,
                    "emotion": cell.emotion,
                    "confidence": cell.confidence,
                    "burnout_risk": cell.burnout_risk,
                    "emoji": cell.emoji,
                    "color": cell.color,
                    "last_updated": cell.last_updated,
                    "age_seconds": current_time - cell.last_updated
                }
        
        # Get team summary
        team_summary = self.team_data.get(team_id, {})
        
        return {
            "grid_size": self.grid_size,
            "grid": grid,
            "team_id": team_id,
            "team_summary": team_summary,
            "total_speakers": len(self.cells),
            "last_updated": current_time,
            "update_interval": self.update_interval
        }
    
    def _cleanup_old_cells(self, current_time: float):
        """Remove cells that haven't been updated recently"""
        threshold = current_time - (self.update_interval * 10)  # 10x update interval
        
        old_speakers = [
            speaker_id for speaker_id, cell in self.cells.items()
            if cell.last_updated < threshold
        ]
        
        for speaker_id in old_speakers:
            del self.cells[speaker_id]
            logger.info(f"Removed old cell for speaker: {speaker_id}")
    
    def get_team_comparison(self, team_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple teams' moods
        
        Args:
            team_ids: List of team identifiers
            
        Returns:
            Dict containing team comparison data
        """
        comparison = {
            "teams": {},
            "comparison_metrics": {},
            "timestamp": time.time()
        }
        
        for team_id in team_ids:
            if team_id in self.team_data:
                team_data = self.team_data[team_id]
                comparison["teams"][team_id] = {
                    "speaker_count": len(team_data.get("speakers", {})),
                    "dominant_emotion": team_data.get("overall_mood", {}).get("dominant_emotion", "unknown"),
                    "avg_burnout_risk": team_data.get("overall_mood", {}).get("avg_burnout_risk", 0.0),
                    "burnout_alert_count": team_data.get("overall_mood", {}).get("burnout_alert_count", 0),
                    "last_updated": team_data.get("last_updated", 0)
                }
        
        # Calculate comparison metrics
        if len(comparison["teams"]) > 1:
            burnout_risks = [
                team_data["avg_burnout_risk"] 
                for team_data in comparison["teams"].values()
            ]
            
            comparison["comparison_metrics"] = {
                "highest_burnout_risk": max(burnout_risks) if burnout_risks else 0.0,
                "lowest_burnout_risk": min(burnout_risks) if burnout_risks else 0.0,
                "avg_burnout_risk": sum(burnout_risks) / len(burnout_risks) if burnout_risks else 0.0,
                "total_burnout_alerts": sum(
                    team_data["burnout_alert_count"] 
                    for team_data in comparison["teams"].values()
                )
            }
        
        return comparison
    
    def generate_mood_report(self, team_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Generate a mood report for a team
        
        Args:
            team_id: Team identifier
            hours: Time period to analyze
            
        Returns:
            Dict containing mood report
        """
        if team_id not in self.team_data:
            return {"error": f"No data found for team {team_id}"}
        
        team = self.team_data[team_id]
        current_time = time.time()
        
        # Calculate time-based metrics
        recent_speakers = {
            speaker_id: data for speaker_id, data in team["speakers"].items()
            if current_time - data["last_updated"] <= hours * 3600
        }
        
        if not recent_speakers:
            return {"error": f"No recent data for team {team_id}"}
        
        # Analyze trends
        emotions = [data["emotion"] for data in recent_speakers.values()]
        burnout_risks = [data["burnout_risk"] for data in recent_speakers.values()]
        
        emotion_distribution = {}
        for emotion in emotions:
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
        
        # Generate insights
        insights = []
        
        if emotion_distribution.get("burned_out", 0) > len(recent_speakers) * 0.3:
            insights.append("High burnout detected - consider team wellness check")
        
        if emotion_distribution.get("energetic", 0) > len(recent_speakers) * 0.5:
            insights.append("Team showing high energy - good momentum")
        
        if sum(burnout_risks) / len(burnout_risks) > 0.6:
            insights.append("Elevated burnout risk across team")
        
        return {
            "team_id": team_id,
            "time_period_hours": hours,
            "active_speakers": len(recent_speakers),
            "emotion_distribution": emotion_distribution,
            "avg_burnout_risk": sum(burnout_risks) / len(burnout_risks),
            "insights": insights,
            "recommendations": self._generate_recommendations(emotion_distribution, burnout_risks),
            "generated_at": current_time
        }
    
    def _generate_recommendations(self, 
                                 emotion_distribution: Dict[str, int], 
                                 burnout_risks: List[float]) -> List[str]:
        """Generate recommendations based on mood analysis"""
        recommendations = []
        
        # Burnout recommendations
        avg_burnout = sum(burnout_risks) / len(burnout_risks) if burnout_risks else 0
        if avg_burnout > 0.7:
            recommendations.append("Schedule team wellness session")
            recommendations.append("Consider workload redistribution")
        
        # Emotion-based recommendations
        if emotion_distribution.get("frustrated", 0) > emotion_distribution.get("calm", 0):
            recommendations.append("Address team frustrations in next meeting")
        
        if emotion_distribution.get("anxious", 0) > 0:
            recommendations.append("Provide clear communication about priorities")
        
        if emotion_distribution.get("energetic", 0) > 0:
            recommendations.append("Leverage team energy for challenging tasks")
        
        return recommendations


def test_mood_map():
    """Test the dynamic mood map"""
    print("🗺️ TESTING DYNAMIC MOOD MAP")
    print("=" * 40)
    
    # Initialize mood map
    mood_map = DynamicMoodMap(grid_size=(4, 4))
    
    # Simulate team mood updates
    print("Updating team moods...")
    
    # Team A updates
    mood_map.update_speaker_mood("alice", "energetic", 0.85, 0.1, "team_a")
    mood_map.update_speaker_mood("bob", "calm", 0.72, 0.2, "team_a")
    mood_map.update_speaker_mood("charlie", "frustrated", 0.68, 0.8, "team_a")
    
    # Team B updates
    mood_map.update_speaker_mood("diana", "anxious", 0.91, 0.6, "team_b")
    mood_map.update_speaker_mood("eve", "burned_out", 0.95, 0.9, "team_b")
    mood_map.update_speaker_mood("frank", "calm", 0.78, 0.3, "team_b")
    
    # Get mood map data
    print("\nGetting mood map data...")
    team_a_data = mood_map.get_mood_map_data("team_a")
    print(f"Team A Mood Map: {json.dumps(team_a_data, indent=2)}")
    
    # Get team comparison
    print("\nGetting team comparison...")
    comparison = mood_map.get_team_comparison(["team_a", "team_b"])
    print(f"Team Comparison: {json.dumps(comparison, indent=2)}")
    
    # Generate mood report
    print("\nGenerating mood report...")
    report = mood_map.generate_mood_report("team_a", hours=1)
    print(f"Mood Report: {json.dumps(report, indent=2)}")
    
    print("✅ Mood map test completed!")


if __name__ == "__main__":
    test_mood_map() 