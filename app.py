"""
Flask Web Application for Zoom Integration
Handles webhooks and provides dashboard for sentiment analysis results
"""

from flask import Flask, request, jsonify, render_template_string
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import sqlite3
from pathlib import Path

from zoom_integration import ZoomIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Zoom integration
zoom_integration = ZoomIntegration()

# Database setup
DB_PATH = "zoom_analysis.db"

def init_database():
    """Initialize SQLite database for storing analysis results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zoom_meeting_id TEXT UNIQUE,
            meeting_topic TEXT,
            start_time TEXT,
            end_time TEXT,
            duration INTEGER,
            participants_count INTEGER,
            recording_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER,
            zoom_meeting_id TEXT,
            sentiment_score REAL,
            stress_level REAL,
            confidence_score REAL,
            emotion_prediction TEXT,
            analysis_data TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meeting_id) REFERENCES meetings (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS webhook_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            meeting_id TEXT,
            payload TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def store_analysis_result(meeting_id: str, analysis_data: Dict):
    """Store analysis result in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Store meeting info if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO meetings (zoom_meeting_id, meeting_topic, start_time)
            VALUES (?, ?, ?)
        ''', (meeting_id, analysis_data.get('meeting_topic', 'Unknown'), 
              datetime.now().isoformat()))
        
        # Get meeting record ID
        cursor.execute('SELECT id FROM meetings WHERE zoom_meeting_id = ?', (meeting_id,))
        meeting_record = cursor.fetchone()
        
        if meeting_record:
            meeting_db_id = meeting_record[0]
            
            # Store analysis result
            cursor.execute('''
                INSERT INTO analysis_results 
                (meeting_id, zoom_meeting_id, sentiment_score, stress_level, 
                 confidence_score, emotion_prediction, analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                meeting_db_id,
                meeting_id,
                analysis_data.get('sentiment_score', 0.0),
                analysis_data.get('stress_level', 0.0),
                analysis_data.get('confidence_score', 0.0),
                analysis_data.get('emotion_prediction', 'unknown'),
                json.dumps(analysis_data)
            ))
            
            conn.commit()
            logger.info(f"Analysis result stored for meeting {meeting_id}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to store analysis result: {e}")

def store_webhook_event(event_type: str, meeting_id: str, payload: Dict):
    """Store webhook event in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO webhook_events (event_type, meeting_id, payload)
            VALUES (?, ?, ?)
        ''', (event_type, meeting_id, json.dumps(payload)))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to store webhook event: {e}")

@app.route('/webhook/zoom', methods=['POST'])
def zoom_webhook():
    """
    Handle Zoom webhook events
    """
    try:
        # Get request data
        payload = request.get_data(as_text=True)
        signature = request.headers.get('X-Zoom-Signature')
        
        # Parse timestamp safely
        signature_256 = request.headers.get('X-Zoom-Signature-256', '')
        timestamp = ''
        if signature_256 and ',' in signature_256:
            timestamp = signature_256.split(',')[0].split('=')[1]
        elif signature_256 and '=' in signature_256:
            timestamp = signature_256.split('=')[1]
        
        # Handle Zoom challenge-response verification
        if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded':
            # This is a challenge request from Zoom
            challenge = request.form.get('challenge')
            if challenge:
                logger.info("Received Zoom challenge request")
                return challenge, 200
        
        # Verify webhook signature
        if not zoom_integration.verify_webhook_signature(payload, signature, timestamp):
            logger.warning("Invalid webhook signature")
            return jsonify({"error": "Invalid signature"}), 401
        
        # Parse JSON payload
        event_data = json.loads(payload)
        event_type = event_data.get('event')
        meeting_id = event_data.get('payload', {}).get('object', {}).get('id')
        
        # Store webhook event
        store_webhook_event(event_type, meeting_id, event_data)
        
        # Process webhook
        result = zoom_integration.handle_webhook(event_data)
        
        # Store analysis result if successful
        if result.get('status') == 'success' and 'analysis_result' in result:
            store_analysis_result(meeting_id, result['analysis_result'])
        
        logger.info(f"Webhook processed: {event_type} for meeting {meeting_id}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "zoom_integration": "active"
    })

@app.route('/meetings', methods=['GET'])
def list_meetings():
    """List all analyzed meetings"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.zoom_meeting_id, m.meeting_topic, m.start_time, m.duration,
                   COUNT(a.id) as analysis_count,
                   AVG(a.sentiment_score) as avg_sentiment,
                   AVG(a.stress_level) as avg_stress
            FROM meetings m
            LEFT JOIN analysis_results a ON m.id = a.meeting_id
            GROUP BY m.id
            ORDER BY m.created_at DESC
        ''')
        
        meetings = []
        for row in cursor.fetchall():
            meetings.append({
                "meeting_id": row[0],
                "topic": row[1],
                "start_time": row[2],
                "duration": row[3],
                "analysis_count": row[4],
                "avg_sentiment": row[5],
                "avg_stress": row[6]
            })
        
        conn.close()
        return jsonify({"meetings": meetings})
        
    except Exception as e:
        logger.error(f"Failed to list meetings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/meeting/<meeting_id>', methods=['GET'])
def get_meeting_details(meeting_id):
    """Get detailed analysis for a specific meeting"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get meeting info
        cursor.execute('''
            SELECT * FROM meetings WHERE zoom_meeting_id = ?
        ''', (meeting_id,))
        
        meeting = cursor.fetchone()
        if not meeting:
            return jsonify({"error": "Meeting not found"}), 404
        
        # Get analysis results
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE zoom_meeting_id = ?
            ORDER BY processed_at DESC
        ''', (meeting_id,))
        
        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                "id": row[0],
                "sentiment_score": row[3],
                "stress_level": row[4],
                "confidence_score": row[5],
                "emotion_prediction": row[6],
                "analysis_data": json.loads(row[7]),
                "processed_at": row[8]
            })
        
        conn.close()
        
        return jsonify({
            "meeting": {
                "id": meeting[1],
                "topic": meeting[2],
                "start_time": meeting[3],
                "end_time": meeting[4],
                "duration": meeting[5],
                "participants_count": meeting[6],
                "recording_status": meeting[7]
            },
            "analyses": analyses
        })
        
    except Exception as e:
        logger.error(f"Failed to get meeting details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def dashboard():
    """Simple dashboard HTML"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speaking Feedback Tool - Zoom Integration</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .meeting { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 3px; }
            .status { padding: 5px 10px; border-radius: 3px; color: white; }
            .success { background: #28a745; }
            .warning { background: #ffc107; }
            .error { background: #dc3545; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; }
            .primary { background: #007bff; color: white; }
            .secondary { background: #6c757d; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎤 Speaking Feedback Tool</h1>
            <p>Zoom Integration Dashboard</p>
        </div>
        
        <div class="section">
            <h2>📊 System Status</h2>
            <div id="status">Loading...</div>
        </div>
        
        <div class="section">
            <h2>📋 Recent Meetings</h2>
            <button class="primary" onclick="loadMeetings()">Refresh Meetings</button>
            <div id="meetings">Loading...</div>
        </div>
        
        <div class="section">
            <h2>🔧 Webhook Endpoint</h2>
            <p><strong>URL:</strong> <code>/webhook/zoom</code></p>
            <p><strong>Method:</strong> POST</p>
            <p><strong>Status:</strong> <span class="status success">Active</span></p>
        </div>
        
        <script>
            // Load system status
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerHTML = `
                        <p><strong>Status:</strong> <span class="status success">${data.status}</span></p>
                        <p><strong>Last Updated:</strong> ${data.timestamp}</p>
                        <p><strong>Zoom Integration:</strong> <span class="status success">${data.zoom_integration}</span></p>
                    `;
                })
                .catch(error => {
                    document.getElementById('status').innerHTML = `
                        <p><strong>Status:</strong> <span class="status error">Error</span></p>
                        <p>Error: ${error.message}</p>
                    `;
                });
            
            // Load meetings
            function loadMeetings() {
                fetch('/meetings')
                    .then(response => response.json())
                    .then(data => {
                        if (data.meetings && data.meetings.length > 0) {
                            const meetingsHtml = data.meetings.map(meeting => `
                                <div class="meeting">
                                    <h3>${meeting.topic || 'Untitled Meeting'}</h3>
                                    <p><strong>ID:</strong> ${meeting.meeting_id}</p>
                                    <p><strong>Start Time:</strong> ${meeting.start_time}</p>
                                    <p><strong>Duration:</strong> ${meeting.duration || 'Unknown'} minutes</p>
                                    <p><strong>Analyses:</strong> ${meeting.analysis_count}</p>
                                    <p><strong>Avg Sentiment:</strong> ${meeting.avg_sentiment ? meeting.avg_sentiment.toFixed(2) : 'N/A'}</p>
                                    <p><strong>Avg Stress:</strong> ${meeting.avg_stress ? meeting.avg_stress.toFixed(2) : 'N/A'}</p>
                                </div>
                            `).join('');
                            document.getElementById('meetings').innerHTML = meetingsHtml;
                        } else {
                            document.getElementById('meetings').innerHTML = '<p>No meetings found.</p>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('meetings').innerHTML = `<p>Error loading meetings: ${error.message}</p>`;
                    });
            }
            
            // Load meetings on page load
            loadMeetings();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/oauth/callback', methods=['GET'])
def oauth_callback():
    """Handle OAuth callback from Zoom"""
    try:
        # Get authorization code from Zoom
        code = request.args.get('code')
        
        if not code:
            return jsonify({"error": "No authorization code received"}), 400
        
        logger.info(f"Received OAuth callback with code: {code[:10]}...")
        
        # In a real implementation, you would exchange this code for an access token
        # For now, we'll just log it and return success
        return jsonify({
            "status": "success",
            "message": "OAuth callback received successfully",
            "code_received": True
        }), 200
        
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/meetings/<meeting_id>/analysis', methods=['GET'])
def get_meeting_analysis(meeting_id):
    """Get analysis results for a specific meeting"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE zoom_meeting_id = ?
            ORDER BY processed_at DESC
        ''', (meeting_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "sentiment_score": row[3],
                "stress_level": row[4],
                "confidence_score": row[5],
                "emotion_prediction": row[6],
                "analysis_data": json.loads(row[7]),
                "processed_at": row[8]
            })
        
        conn.close()
        return jsonify({"meeting_id": meeting_id, "analyses": results})
        
    except Exception as e:
        logger.error(f"Failed to get meeting analysis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Get port from environment or default to 5001
    port = int(os.environ.get('PORT', 5001))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 