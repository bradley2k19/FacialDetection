""" Flask Backend for Depression Detection Web Application ====================================================== This serves the web interface and handles video upload and real-time predictions. """
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import os
from datetime import datetime
import cv2
from pathlib import Path
import threading
import shutil

# Import our custom modules
from .depression_model import DepressionPredictor
from .openface_processor import OpenFaceProcessor  # Using your existing processor

# Get the correct paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / 'frontend' / 'templates'
STATIC_DIR = BASE_DIR / 'frontend' / 'static'

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    engineio_logger=False
)

print(f"📁 Template folder: {TEMPLATE_DIR}")
print(f"📁 Static folder: {STATIC_DIR}")

# Paths
TEMP_DIR = Path('temp_openface')
UPLOADS_DIR = Path('uploads')
OPENFACE_EXE = r'C:\Users\Augustin Bradley\Downloads\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'
OPENFACE_MODEL = 'model/main_clnf_general.txt'

# Create directories
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize model and processor
print("🔄 Loading model and OpenFace...")
predictor = DepressionPredictor(
    model_path='best_depression_model.keras',
    scaler_path='feature_scaler.pkl'
)
openface = OpenFaceProcessor(
    openface_path=OPENFACE_EXE,
    temp_dir=str(TEMP_DIR)
)
print("✅ Model and OpenFace loaded successfully!")

# Store session data
session_data = {}
prediction_history = {}

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
    })

@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    client_id = request.sid
    session_data[client_id] = {
        'chunks': {},
        'total_chunks': 0
    }
    prediction_history[client_id] = []
    print(f"✅ Client connected: {client_id}")
    print(f"   Ready to receive: video_chunk, analyze events")
    emit('response', {'data': 'Connected to server'}, broadcast=False)

@socketio.on_error_default
def default_error_handler(e):
    print(f"❌ Socket.IO error: {str(e)}")
    import traceback
    traceback.print_exc()

# Test handler to see if events are arriving
@socketio.on('*')
def catch_all(event, *args):
    """Catch all events for debugging."""
    print(f"📨 Event received: {event}")
    if event == 'video_chunk':
        print(f"   Args: {args}")

@socketio.on('video_chunk')
def handle_video_chunk(data):
    """Receive video chunks from frontend and store them temporarily."""
    client_id = request.sid
    
    try:
        chunk_id = data.get('chunk_id')
        chunk_data = data.get('chunk_data')
        is_last = data.get('is_last', False)
        
        if client_id not in session_data:
            session_data[client_id] = {
                'chunks': {},
                'total_chunks': 0
            }
        
        # Store chunk
        if chunk_data:
            session_data[client_id]['chunks'][chunk_id] = chunk_data
            print(f"📦 Chunk {chunk_id} stored for client {client_id}")
        
        # If last chunk, send confirmation
        if is_last:
            session_data[client_id]['total_chunks'] = chunk_id + 1
            print(f"✅ All chunks received for {client_id} - sending chunks_received event")
            socketio.emit('chunks_received', {'status': 'ready'}, to=client_id)
        
        return {'status': 'received', 'chunk_id': chunk_id}
        
    except Exception as e:
        print(f"❌ Error in video_chunk: {str(e)}")
        return {'status': 'error'}

@socketio.on('analyze')
def handle_analyze(data):
    """Process the received video file."""
    client_id = request.sid
    try:
        total_chunks = data.get('total_chunks')
        
        print(f"🎬 Starting video processing for {total_chunks} chunks")
        
        # Run processing in background thread
        thread = threading.Thread(
            target=process_video,
            args=(client_id, total_chunks)
        )
        thread.daemon = True
        thread.start()
        
        socketio.emit('processing_started', {'status': 'Video processing started'}, to=client_id)
    except Exception as e:
        print(f"❌ Error in analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f'Error starting analysis: {str(e)}'}, to=client_id)

def process_video(client_id, total_chunks):
    """Process video: reassemble chunks and analyze with OpenFace."""
    try:
        print(f"\n🎬 Processing video for client: {client_id}")
        
        # Reassemble video from chunks
        print("🔗 Reassembling video from chunks...")
        video_data = bytearray()
        
        for i in range(total_chunks):
            if i not in session_data[client_id]['chunks']:
                print(f"⚠️ Chunk {i} missing!")
                continue
                
            chunk = session_data[client_id]['chunks'][i]
            
            if isinstance(chunk, str):
                try:
                    chunk_bytes = bytes.fromhex(chunk)
                    video_data.extend(chunk_bytes)
                except Exception as e:
                    print(f"✗ Failed to convert chunk {i}: {e}")
        
        # Save video file
        video_path = TEMP_DIR / f"video_{client_id}.webm"
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        file_size = len(video_data)
        print(f"💾 Video saved: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Video file is empty")
        
        # Extract frames
        socketio.emit('processing_update', {'status': 'Extracting frames from video...'}, to=client_id)
        frames = extract_frames(str(video_path))
        
        if not frames:
            raise Exception("No frames extracted")
        
        print(f"📹 Extracted {len(frames)} frames")
        
        # Extract features
        socketio.emit('processing_update', {'status': f'Analyzing {len(frames)} frames with OpenFace...'}, to=client_id)
        features = extract_features_from_frames(frames)
        
        if features is None or len(features) == 0:
            raise Exception("Feature extraction failed")
        
        print(f"✅ Features extracted: {features.shape}")
        
        # Make prediction
        socketio.emit('processing_update', {'status': 'Making prediction...'}, to=client_id)
        probability, explanation, feature_stats = predictor.predict(features)
        
        # Update history
        if client_id not in prediction_history:
            prediction_history[client_id] = []
        
        prediction_history[client_id].append({
            'probability': float(probability),
            'timestamp': datetime.now().isoformat(),
            'explanation': explanation,
            'results': format_frame_results(features),
            'total_frames': len(frames),
            'frames_analyzed': len(features),
            'top_action_units': feature_stats.get('top_action_units', [])
        })
        
        if len(prediction_history[client_id]) > 50:
            prediction_history[client_id] = prediction_history[client_id][-50:]
        
        # Send result
        result = {
            'status': 'success',
            'probability': float(probability),
            'risk_level': 'HIGH' if probability >= 0.7 else 'MODERATE' if probability >= 0.5 else 'LOW',
            'explanation': explanation,
            'feature_stats': feature_stats,
            'total_frames': len(frames),
            'frames_analyzed': len(features),
            'history': prediction_history[client_id],
            'timestamp': datetime.now().isoformat(),
            'results': format_frame_results(features)
        }
        
        socketio.emit('analysis_complete', result, to=client_id)
        print(f"✅ Analysis complete: {probability*100:.2f}%")
        
        cleanup_session(client_id)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        socketio.emit('error', {
            'message': f'Processing error: {str(e)}',
            'error_type': 'processing'
        }, to=client_id)
        cleanup_session(client_id)

def extract_frames(video_path, max_frames=300):
    """Extract frames from video file with a reasonable limit."""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            # Limit to 300 frames (10 seconds at 30fps)
            if max_frames and frame_count >= max_frames:
                print(f"⚠️ Reached frame limit of {max_frames}")
                break
        
        cap.release()
        print(f"✅ Successfully extracted {len(frames)} frames")
        return frames
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return []

def extract_features_from_frames(frames):
    """Extract features from frames using OpenFaceProcessor with frame skipping."""
    try:

        FRAME_SKIP = 1
        sampled_frames = frames[::FRAME_SKIP]
        
        print(f"📹 Processing {len(sampled_frames)} frames (skipping every {FRAME_SKIP}th frame from {len(frames)} total)")
        
        # Convert RGB frames to BGR for OpenFaceProcessor
        bgr_frames = []
        for frame in sampled_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr_frames.append(frame_bgr)
        
        # Use the OpenFaceProcessor to extract features
        print(f"🔍 Running OpenFace on sampled video...")
        features = openface.extract_features(bgr_frames)
        
        if features is None:
            print("❌ OpenFaceProcessor returned None")
            return None
        
        print(f"✅ OpenFace analysis completed: {features.shape}")
        return features
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_openface_csv(csv_path):
    """Parse OpenFace CSV output and extract relevant features."""
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return []
        
        headers = lines[0].strip().split(',')
        features_list = []
        
        # Find relevant column indices
        success_idx = None
        au_indices = []
        
        for idx, header in enumerate(headers):
            header = header.strip()
            if 'confidence' in header.lower() or 'success' in header.lower():
                success_idx = idx
            if header.startswith('AU'):
                au_indices.append(idx)
        
        # Parse feature rows
        for line in lines[1:]:
            values = line.strip().split(',')
            if not values or not values[0]:
                continue
            
            try:
                # Extract success/confidence
                confidence = 1.0
                if success_idx is not None and success_idx < len(values):
                    confidence = float(values[success_idx])
                
                # Extract AU intensities
                aus = []
                for au_idx in au_indices:
                    if au_idx < len(values):
                        try:
                            au_val = float(values[au_idx])
                            aus.append(au_val)
                        except ValueError:
                            aus.append(0.0)
                
                if aus:
                    features_list.append(np.array(aus))
            except Exception as e:
                continue
        
        if features_list:
            return np.array(features_list)
        return None
    except Exception as e:
        print(f"❌ Error parsing CSV: {e}")
        return None

def format_frame_results(features):
    """Format frame-by-frame results for frontend display."""
    results = []
    
    if features is None or len(features) == 0:
        return results
    
    # AU features are first 24 columns, gaze 8, pose 6
    au_count = 24
    
    for frame_idx, frame_features in enumerate(features):
        frame_data = {
            'frame': frame_idx + 1,
            'action_units': {},
            'confidence': float(frame_features[0]) if len(frame_features) > 0 else 1.0
        }
        
        # Extract AU intensities (first 24 features)
        for au_idx in range(min(au_count, len(frame_features))):
            value = frame_features[au_idx]
            if value > 0.01:  # Only include AUs with meaningful intensity
                au_name = f"AU{au_idx + 1:02d}"
                frame_data['action_units'][au_name] = float(value)
        
        results.append(frame_data)
    
    return results

def cleanup_session(client_id):
    """Clean up temporary files for a session."""
    try:
        # Remove video files
        for pattern in ['video_*.webm', 'video_*.avi', 'temp_openface_*.avi']:
            for f in TEMP_DIR.glob(pattern):
                try:
                    if client_id in str(f) or f.stat().st_size < 1024 * 1024:  # Clean old temp files
                        os.remove(f)
                except:
                    pass
        
        # Remove output directories
        for d in TEMP_DIR.glob(f"openface_output_*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        
        # Clear session chunks
        if client_id in session_data:
            session_data[client_id]['chunks'].clear()
        
        print(f"🧹 Cleaned up session: {client_id}")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

@app.route('/download_results')
def download_results():
    """Download the latest analysis results as CSV."""
    client_id = request.args.get('client_id')
    result_type = request.args.get('type', 'summary')  # 'summary' or 'detailed'
    
    if client_id not in prediction_history or not prediction_history[client_id]:
        return jsonify({'error': 'No predictions found'}), 404
    
    if result_type == 'summary':
        # Summary CSV - one row per analysis
        csv_lines = ['timestamp,probability,risk_level,explanation,total_frames,frames_analyzed,top_action_units']
        
        for pred in prediction_history[client_id]:
            prob = pred['probability']
            risk = 'HIGH' if prob >= 0.7 else 'MODERATE' if prob >= 0.5 else 'LOW'
            explanation = pred.get('explanation', 'Analysis completed').replace('"', '""')
            timestamp = pred['timestamp']
            total_frames = pred.get('total_frames', 0)
            frames_analyzed = pred.get('frames_analyzed', 0)
            
            # Get top action units
            top_aus = pred.get('top_action_units', [])
            au_string = ', '.join([f"{au['name']} ({au['intensity']:.2f})" for au in top_aus])
            
            csv_lines.append(f'{timestamp},{prob:.4f},{risk},"{explanation}",{total_frames},{frames_analyzed},"{au_string}"')
        
        csv_content = '\n'.join(csv_lines)
        filename = 'depression_analysis_summary.csv'
    
    elif result_type == 'detailed':
        # Detailed CSV - frame-by-frame results for the latest analysis
        if not prediction_history[client_id]:
            return jsonify({'error': 'No detailed results available'}), 404
        
        latest_pred = prediction_history[client_id][-1]
        results = latest_pred.get('results', [])
        
        if not results:
            return jsonify({'error': 'No frame results available'}), 404
        
        # Create header with all possible AUs
        all_aus = set()
        for frame_result in results:
            if 'action_units' in frame_result:
                all_aus.update(frame_result['action_units'].keys())
        
        all_aus = sorted(list(all_aus))
        header = ['frame', 'confidence'] + all_aus
        csv_lines = [','.join(header)]
        
        # Add rows
        for frame_result in results:
            row = [str(frame_result.get('frame', '')), 
                   str(frame_result.get('confidence', ''))]
            
            for au in all_aus:
                intensity = frame_result.get('action_units', {}).get(au, '')
                row.append(str(intensity) if intensity != '' else '0')
            
            csv_lines.append(','.join(row))
        
        csv_content = '\n'.join(csv_lines)
        filename = 'depression_analysis_detailed.csv'
    
    else:
        return jsonify({'error': 'Invalid result type'}), 400
    
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename={filename}'
    }

@app.route('/download_log')
def download_log():
    """Download prediction history as CSV."""
    client_id = request.args.get('client_id')
    
    if client_id not in prediction_history or not prediction_history[client_id]:
        return jsonify({'error': 'No predictions found'}), 404
    
    # Create CSV content
    csv_lines = ['timestamp,probability,risk_level']
    for pred in prediction_history[client_id]:
        prob = pred['probability']
        risk = 'HIGH' if prob >= 0.7 else 'MODERATE' if prob >= 0.5 else 'LOW'
        csv_lines.append(f"{pred['timestamp']},{prob:.4f},{risk}")
    
    csv_content = '\n'.join(csv_lines)
    
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=predictions.csv'
    }

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 DEPRESSION DETECTION WEB SERVER")
    print("="*70)
    print("\n📱 Access the application at:")
    print(" Local: http://localhost:5000")
    print(" Network: http://YOUR_IP:5000")
    print("\n⌨️ Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
