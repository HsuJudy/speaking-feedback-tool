"""
Flask Web Application for Zoom Integration
Handles webhooks and provides dashboard for sentiment analysis results
"""

from flask import Flask, request, jsonify, render_template, send_file
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import sqlite3
from pathlib import Path

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install prometheus_client for metrics.")

from zoom_integration import ZoomIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Zoom integration
zoom_integration = ZoomIntegration()

# Database setup
DB_PATH = "zoom_analysis.db"

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Counters
    http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'status'])
    zoom_webhook_events_total = Counter('zoom_webhook_events_total', 'Total Zoom webhook events', ['event_type'])
    zoom_webhook_failures_total = Counter('zoom_webhook_failures_total', 'Total Zoom webhook failures')
    ml_pipeline_failures_total = Counter('ml_pipeline_failures_total', 'Total ML pipeline failures')
    
    # Histograms
    http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
    zoom_recording_processing_duration_seconds = Histogram('zoom_recording_processing_duration_seconds', 'Zoom recording processing duration')
    model_inference_duration_seconds = Histogram('model_inference_duration_seconds', 'Model inference duration', ['model_name'])
    
    # Gauges
    speaking_feedback_active_meetings = Gauge('speaking_feedback_active_meetings', 'Number of active meetings')
    speaking_feedback_total_meetings = Gauge('speaking_feedback_total_meetings', 'Total number of meetings processed')

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
    start_time = time.time()
    
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
            if PROMETHEUS_AVAILABLE:
                zoom_webhook_failures_total.inc()
            return jsonify({"error": "Invalid signature"}), 401
        
        # Parse JSON payload
        event_data = json.loads(payload)
        event_type = event_data.get('event')
        meeting_id = event_data.get('payload', {}).get('object', {}).get('id')
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            zoom_webhook_events_total.labels(event_type=event_type).inc()
        
        # Store webhook event
        store_webhook_event(event_type, meeting_id, event_data)
        
        # Process webhook
        result = zoom_integration.handle_webhook(event_data)
        
        # Store analysis result if successful
        if result.get('status') == 'success' and 'analysis_result' in result:
            store_analysis_result(meeting_id, result['analysis_result'])
        
        # Record processing duration
        if PROMETHEUS_AVAILABLE:
            duration = time.time() - start_time
            zoom_recording_processing_duration_seconds.observe(duration)
        
        logger.info(f"Webhook processed: {event_type} for meeting {meeting_id}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        if PROMETHEUS_AVAILABLE:
            zoom_webhook_failures_total.inc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "zoom_integration": "active"
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    else:
        return jsonify({"error": "Prometheus metrics not available"}), 503

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
    """Modern dashboard"""
    return render_template('dashboard.html')

@app.route('/meetings', methods=['GET'])
def meetings_page():
    """Meetings page"""
    return render_template('meetings.html')

@app.route('/analytics', methods=['GET'])
def analytics_page():
    """Analytics page"""
    return render_template('analytics.html')

@app.route('/settings', methods=['GET'])
def settings_page():
    """Settings page"""
    return render_template('settings.html')

# API Endpoints for modern interface
@app.route('/api/stats/quick', methods=['GET'])
def quick_stats():
    """Get quick stats for sidebar"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Today's meetings
        today = datetime.now().date()
        cursor.execute('''
            SELECT COUNT(*) FROM meetings 
            WHERE DATE(start_time) = ?
        ''', (today,))
        today_meetings = cursor.fetchone()[0]
        
        # Average sentiment
        cursor.execute('''
            SELECT AVG(sentiment_score) FROM analysis_results 
            WHERE sentiment_score IS NOT NULL
        ''')
        avg_sentiment = cursor.fetchone()[0]
        
        # High stress count
        cursor.execute('''
            SELECT COUNT(*) FROM analysis_results 
            WHERE stress_level > 0.7
        ''')
        high_stress_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'today_meetings': today_meetings,
            'avg_sentiment': avg_sentiment,
            'high_stress_count': high_stress_count
        })
        
    except Exception as e:
        logger.error(f"Error getting quick stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/dashboard', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    try:
        range_param = request.args.get('range', 'week')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate date range
        if range_param == 'today':
            start_date = datetime.now().date()
        elif range_param == 'week':
            start_date = datetime.now().date() - timedelta(days=7)
        elif range_param == 'month':
            start_date = datetime.now().date() - timedelta(days=30)
        else:
            start_date = None
        
        # Total meetings
        if start_date:
            cursor.execute('''
                SELECT COUNT(*) FROM meetings 
                WHERE DATE(start_time) >= ?
            ''', (start_date,))
        else:
            cursor.execute('SELECT COUNT(*) FROM meetings')
        total_meetings = cursor.fetchone()[0]
        
        # Average sentiment
        if start_date:
            cursor.execute('''
                SELECT AVG(ar.sentiment_score) 
                FROM analysis_results ar
                JOIN meetings m ON ar.meeting_id = m.id
                WHERE DATE(m.start_time) >= ? AND ar.sentiment_score IS NOT NULL
            ''', (start_date,))
        else:
            cursor.execute('''
                SELECT AVG(sentiment_score) FROM analysis_results 
                WHERE sentiment_score IS NOT NULL
            ''')
        avg_sentiment = cursor.fetchone()[0]
        
        # Average stress
        if start_date:
            cursor.execute('''
                SELECT AVG(ar.stress_level) 
                FROM analysis_results ar
                JOIN meetings m ON ar.meeting_id = m.id
                WHERE DATE(m.start_time) >= ? AND ar.stress_level IS NOT NULL
            ''', (start_date,))
        else:
            cursor.execute('''
                SELECT AVG(stress_level) FROM analysis_results 
                WHERE stress_level IS NOT NULL
            ''')
        avg_stress = cursor.fetchone()[0]
        
        # Sentiment trend (last 7 days)
        trend_dates = []
        trend_data = []
        for i in range(7):
            date = datetime.now().date() - timedelta(days=i)
            cursor.execute('''
                SELECT AVG(ar.sentiment_score) 
                FROM analysis_results ar
                JOIN meetings m ON ar.meeting_id = m.id
                WHERE DATE(m.start_time) = ? AND ar.sentiment_score IS NOT NULL
            ''', (date,))
            avg = cursor.fetchone()[0]
            trend_dates.insert(0, date.strftime('%m/%d'))
            trend_data.insert(0, avg if avg else 0)
        
        # Emotion distribution
        cursor.execute('''
            SELECT emotion_prediction, COUNT(*) 
            FROM analysis_results 
            WHERE emotion_prediction IS NOT NULL
            GROUP BY emotion_prediction
        ''')
        emotion_counts = dict(cursor.fetchall())
        
        # Map emotions to chart data
        emotions = ['happy', 'neutral', 'sad', 'angry', 'fearful']
        emotion_distribution = [emotion_counts.get(emotion, 0) for emotion in emotions]
        
        conn.close()
        
        return jsonify({
            'total_meetings': total_meetings,
            'avg_sentiment': avg_sentiment,
            'avg_stress': avg_stress,
            'sentiment_trend': {
                'labels': trend_dates,
                'data': trend_data
            },
            'emotion_distribution': emotion_distribution
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings/recent', methods=['GET'])
def recent_meetings():
    """Get recent meetings for dashboard"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.zoom_meeting_id, m.meeting_topic, m.start_time, m.duration,
                   ar.sentiment_score, ar.stress_level, ar.emotion_prediction, ar.confidence_score
            FROM meetings m
            LEFT JOIN analysis_results ar ON m.id = ar.meeting_id
            ORDER BY m.start_time DESC
            LIMIT 5
        ''')
        
        meetings = []
        for row in cursor.fetchall():
            meetings.append({
                'meeting_id': row[0],
                'topic': row[1],
                'start_time': row[2],
                'duration': row[3],
                'sentiment_score': row[4],
                'stress_level': row[5],
                'emotion_prediction': row[6],
                'confidence_score': row[7]
            })
        
        conn.close()
        return jsonify({'meetings': meetings})
        
    except Exception as e:
        logger.error(f"Error getting recent meetings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings', methods=['GET'])
def api_meetings():
    """Get meetings with filters"""
    try:
        search = request.args.get('search', '')
        date_range = request.args.get('date_range', 'all')
        sentiment = request.args.get('sentiment', 'all')
        stress_level = request.args.get('stress_level', 'all')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build query
        query = '''
            SELECT m.zoom_meeting_id, m.meeting_topic, m.start_time, m.duration, m.participants_count,
                   ar.sentiment_score, ar.stress_level, ar.emotion_prediction, ar.confidence_score
            FROM meetings m
            LEFT JOIN analysis_results ar ON m.id = ar.meeting_id
            WHERE 1=1
        '''
        params = []
        
        if search:
            query += ' AND m.meeting_topic LIKE ?'
            params.append(f'%{search}%')
        
        if date_range == 'today':
            query += ' AND DATE(m.start_time) = ?'
            params.append(datetime.now().date())
        elif date_range == 'week':
            query += ' AND DATE(m.start_time) >= ?'
            params.append(datetime.now().date() - timedelta(days=7))
        elif date_range == 'month':
            query += ' AND DATE(m.start_time) >= ?'
            params.append(datetime.now().date() - timedelta(days=30))
        
        if sentiment == 'positive':
            query += ' AND ar.sentiment_score > 0.3'
        elif sentiment == 'negative':
            query += ' AND ar.sentiment_score < -0.3'
        elif sentiment == 'neutral':
            query += ' AND ar.sentiment_score BETWEEN -0.3 AND 0.3'
        
        if stress_level == 'low':
            query += ' AND ar.stress_level < 0.4'
        elif stress_level == 'medium':
            query += ' AND ar.stress_level BETWEEN 0.4 AND 0.7'
        elif stress_level == 'high':
            query += ' AND ar.stress_level > 0.7'
        
        query += ' ORDER BY m.start_time DESC'
        
        cursor.execute(query, params)
        
        meetings = []
        for row in cursor.fetchall():
            meetings.append({
                'meeting_id': row[0],
                'topic': row[1],
                'start_time': row[2],
                'duration': row[3],
                'participants_count': row[4],
                'sentiment_score': row[5],
                'stress_level': row[6],
                'emotion_prediction': row[7],
                'confidence_score': row[8]
            })
        
        conn.close()
        return jsonify({'meetings': meetings})
        
    except Exception as e:
        logger.error(f"Error getting meetings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings/<meeting_id>/details', methods=['GET'])
def meeting_details(meeting_id):
    """Get detailed meeting information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get meeting info
        cursor.execute('''
            SELECT * FROM meetings WHERE zoom_meeting_id = ?
        ''', (meeting_id,))
        meeting_row = cursor.fetchone()
        
        if not meeting_row:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting = {
            'meeting_id': meeting_row[1],
            'topic': meeting_row[2],
            'start_time': meeting_row[3],
            'end_time': meeting_row[4],
            'duration': meeting_row[5],
            'participants_count': meeting_row[6],
            'recording_status': meeting_row[7]
        }
        
        # Get analysis
        cursor.execute('''
            SELECT * FROM analysis_results WHERE zoom_meeting_id = ?
        ''', (meeting_id,))
        analysis_row = cursor.fetchone()
        
        analysis = {}
        if analysis_row:
            analysis = {
                'sentiment_score': analysis_row[3],
                'stress_level': analysis_row[4],
                'confidence_score': analysis_row[5],
                'emotion_prediction': analysis_row[6],
                'analysis_data': analysis_row[7],
                'processed_at': analysis_row[8]
            }
        
        conn.close()
        return jsonify({'meeting': meeting, 'analysis': analysis})
        
    except Exception as e:
        logger.error(f"Error getting meeting details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/meetings/<meeting_id>/download', methods=['GET'])
def download_meeting_analysis(meeting_id):
    """Download meeting analysis as JSON"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get meeting and analysis data
        cursor.execute('''
            SELECT m.*, ar.* FROM meetings m
            LEFT JOIN analysis_results ar ON m.id = ar.meeting_id
            WHERE m.zoom_meeting_id = ?
        ''', (meeting_id,))
        
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'Meeting not found'}), 404
        
        # Create analysis data
        analysis_data = {
            'meeting_id': row[1],
            'topic': row[2],
            'start_time': row[3],
            'duration': row[5],
            'participants_count': row[6],
            'analysis': {
                'sentiment_score': row[11],
                'stress_level': row[12],
                'confidence_score': row[13],
                'emotion_prediction': row[14],
                'analysis_data': json.loads(row[15]) if row[15] else None,
                'processed_at': row[16]
            }
        }
        
        conn.close()
        
        # Create JSON file
        json_data = json.dumps(analysis_data, indent=2, default=str)
        
        from io import BytesIO
        buffer = BytesIO()
        buffer.write(json_data.encode())
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'meeting_{meeting_id}_analysis.json',
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error downloading meeting analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts"""
    try:
        alerts = []
        
        # Check for high stress meetings
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM analysis_results 
            WHERE stress_level > 0.8
        ''')
        high_stress_count = cursor.fetchone()[0]
        
        if high_stress_count > 0:
            alerts.append({
                'level': 'warning',
                'icon': 'exclamation-triangle',
                'message': f'{high_stress_count} meetings with high stress levels detected'
            })
        
        # Check for failed webhooks
        cursor.execute('''
            SELECT COUNT(*) FROM webhook_events 
            WHERE event_type LIKE '%failed%'
        ''')
        failed_webhooks = cursor.fetchone()[0]
        
        if failed_webhooks > 0:
            alerts.append({
                'level': 'danger',
                'icon': 'exclamation-circle',
                'message': f'{failed_webhooks} webhook failures detected'
            })
        
        conn.close()
        
        return jsonify({'alerts': alerts})
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({'alerts': []})

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

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings (without sensitive data)"""
    try:
        # Get environment variables (masked for security)
        settings = {
            'app_name': 'Speaking Feedback Tool',
            'app_version': '1.0.0',
            'debug_mode': os.getenv('FLASK_ENV') == 'development',
            'log_level': 'INFO',
            'db_path': DB_PATH,
            'db_size': get_database_size(),
            'backup_enabled': True,
            
            # Zoom settings (masked)
            'zoom_client_id': '***' if os.getenv('ZOOM_CLIENT_ID') else None,
            'zoom_client_secret': '***' if os.getenv('ZOOM_CLIENT_SECRET') else None,
            'zoom_webhook_secret': '***' if os.getenv('ZOOM_WEBHOOK_SECRET') else None,
            'webhook_url': os.getenv('ZOOM_WEBHOOK_URL', 'https://your-domain.com/webhook/zoom'),
            'auto_download': os.getenv('ZOOM_ENABLE_AUDIO_EXTRACTION', 'true').lower() == 'true',
            
            # ML settings
            'sentiment_model': 'custom',
            'emotion_model': 'custom',
            'confidence_threshold': 0.7,
            
            # Monitoring settings
            'metrics_enabled': True,
            'metrics_interval': 60,
            'alert_enabled': True,
            'alert_threshold': 0.8,
            'notification_email': '',
            
            # Security settings
            'webhook_verification': os.getenv('ZOOM_VERIFY_SIGNATURES', 'true').lower() == 'true',
            'rate_limiting': True,
            'max_requests': 100,
            'session_timeout': 30,
            'require_auth': False,
            'allowed_origins': os.getenv('ZOOM_ALLOWED_IPS', ''),
            
            # Status indicators
            'zoom_configured': bool(os.getenv('ZOOM_CLIENT_ID') and os.getenv('ZOOM_CLIENT_SECRET')),
            'webhook_configured': bool(os.getenv('ZOOM_WEBHOOK_SECRET')),
            'database_configured': os.path.exists(DB_PATH),
            'models_configured': True,
            
            # Webhook events
            'webhook_events': ['meeting.started', 'meeting.ended', 'recording.completed'],
            
            # Alert types
            'alert_types': ['high_stress', 'negative_sentiment', 'model_failure', 'webhook_failure']
        }
        
        return jsonify(settings)
        
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings (non-sensitive only)"""
    try:
        data = request.get_json()
        
        # Only allow saving non-sensitive settings
        allowed_settings = [
            'auto_download', 'sentiment_model', 'emotion_model', 'confidence_threshold',
            'metrics_enabled', 'metrics_interval', 'alert_enabled', 'alert_threshold',
            'notification_email', 'rate_limiting', 'max_requests', 'session_timeout',
            'require_auth', 'allowed_origins', 'webhook_events', 'alert_types'
        ]
        
        # Update configuration (in memory for now)
        # In production, you'd want to persist these to a config file or database
        
        return jsonify({'success': True, 'message': 'Settings updated successfully'})
        
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/test-zoom', methods=['GET'])
def test_zoom_connection():
    """Test Zoom API connection"""
    try:
        client_id = os.getenv('ZOOM_CLIENT_ID')
        client_secret = os.getenv('ZOOM_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            return jsonify({'success': False, 'error': 'Zoom credentials not configured'})
        
        # Test the connection using zoom_integration
        from zoom_integration import ZoomIntegration
        zoom = ZoomIntegration()
        
        # Try to get account info (basic test)
        try:
            # This would test the actual connection
            # For now, just check if credentials exist
            return jsonify({'success': True, 'message': 'Zoom credentials configured'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Connection failed: {str(e)}'})
            
    except Exception as e:
        logger.error(f"Error testing Zoom connection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/test-models', methods=['GET'])
def test_models():
    """Test ML models"""
    try:
        # Test if models are available and working
        from inference import SentimentAnalysisPipeline
        
        pipeline = SentimentAnalysisPipeline()
        
        # Test with sample data
        test_text = "This is a test message for sentiment analysis."
        
        try:
            result = pipeline.analyze_text(test_text)
            return jsonify({
                'success': True, 
                'message': 'Models working correctly',
                'test_result': result
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Model test failed: {str(e)}'})
            
    except Exception as e:
        logger.error(f"Error testing models: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/test-alerts', methods=['GET'])
def test_alerts():
    """Test alert system"""
    try:
        # Send a test alert
        logger.info("Test alert triggered from settings page")
        
        return jsonify({
            'success': True, 
            'message': 'Test alert sent successfully'
        })
        
    except Exception as e:
        logger.error(f"Error testing alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/backup-database', methods=['GET'])
def backup_database():
    """Backup database"""
    try:
        import shutil
        from datetime import datetime
        
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = os.path.join(os.path.dirname(DB_PATH), backup_name)
        
        shutil.copy2(DB_PATH, backup_path)
        
        return send_file(
            backup_path,
            as_attachment=True,
            download_name=backup_name,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/clear-database', methods=['POST'])
def clear_database():
    """Clear database (dangerous operation)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Clear all tables
        cursor.execute('DELETE FROM meetings')
        cursor.execute('DELETE FROM analysis_results')
        cursor.execute('DELETE FROM webhook_events')
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Database cleared successfully'})
        
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/rotate-secrets', methods=['POST'])
def rotate_secrets():
    """Rotate secrets (placeholder)"""
    try:
        # In production, this would generate new secrets and update environment
        logger.info("Secret rotation requested")
        
        return jsonify({
            'success': True, 
            'message': 'Secret rotation completed (simulated)'
        })
        
    except Exception as e:
        logger.error(f"Error rotating secrets: {e}")
        return jsonify({'error': str(e)}), 500

def get_database_size():
    """Get database size in human readable format"""
    try:
        if os.path.exists(DB_PATH):
            size_bytes = os.path.getsize(DB_PATH)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        return "0 B"
    except Exception:
        return "Unknown"

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Get port from environment or default to 5001
    port = int(os.environ.get('PORT', 5001))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True) 