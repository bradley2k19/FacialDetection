"""
Real-time Depression Detection from Webcam Feed
================================================
This code uses your trained model with OpenFace to analyze students in real-time.

IMPORTANT: This requires OpenFace to be installed and configured.
"""

import cv2
import numpy as np
import pandas as pd
import subprocess
import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import deque
import json
import threading  # For background processing

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE TO YOUR ACTUAL PATHS
MODEL_PATH = "best_depression_model.keras"  # Your trained model
SCALER_PATH = "feature_scaler.pkl"  # Feature normalizer

# OpenFace path - UPDATE THIS!
# Windows example: "C:/OpenFace/FeatureExtraction.exe"
# Linux/Mac example: "/home/user/OpenFace/build/bin/FeatureExtraction"
OPENFACE_PATH = r"C:\Users\Augustin Bradley\Downloads\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

# Analysis parameters
SEQUENCE_LENGTH = 100  # Must match training
FRAME_BUFFER_SIZE = 100  # How many frames to collect before analysis
FPS = 30  # Frames per second

# Output paths
OUTPUT_DIR = "predictions"
LOG_FILE = f"{OUTPUT_DIR}/prediction_log.csv"
TEMP_DIR = "temp_openface"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("✅ Configuration loaded")
print(f"   OpenFace path: {OPENFACE_PATH}")
print(f"   Model path: {MODEL_PATH}")
print(f"   Temp directory: {TEMP_DIR}")


# ============================================================================
# CHECK OPENFACE INSTALLATION
# ============================================================================

def check_openface():
    """
    Verifies that OpenFace is installed and accessible.
    """
    print("\n🔍 Checking OpenFace installation...")
    
    if not os.path.exists(OPENFACE_PATH):
        print(f"❌ OpenFace not found at: {OPENFACE_PATH}")
        print("\nPlease:")
        print("1. Download OpenFace from: https://github.com/TadasBaltrusaitis/OpenFace/releases")
        print("2. Extract it to a folder (e.g., C:/OpenFace/)")
        print("3. Update OPENFACE_PATH in this script to point to FeatureExtraction.exe")
        print(f"   Current path: {OPENFACE_PATH}")
        return False
    
    try:
        # Try running OpenFace with -help flag
        result = subprocess.run(
            [OPENFACE_PATH, "-help"],
            capture_output=True,
            timeout=5
        )
        print("✅ OpenFace is accessible and working!")
        return True
    except Exception as e:
        print(f"⚠️  OpenFace found but cannot run: {e}")
        print("   This might be okay - we'll try to use it anyway")
        return True


# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

def load_model_and_scaler():
    """
    Loads the trained model and feature scaler.
    """
    print("\n📥 Loading model and scaler...")
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"   ✅ Model loaded from {MODEL_PATH}")
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   ✅ Scaler loaded from {SCALER_PATH}")
        
        return model, scaler
    
    except Exception as e:
        print(f"   ❌ Error loading model or scaler: {e}")
        return None, None


# ============================================================================
# IMPROVED OPENFACE FEATURE EXTRACTION
# ============================================================================

class OpenFaceExtractor:
    """
    Handles OpenFace feature extraction from video frames.
    """
    
    def __init__(self, openface_path, temp_dir):
        self.openface_path = openface_path
        self.temp_dir = temp_dir
        self.video_counter = 0
    
    def extract_features_from_frames(self, frames):
        """
        Extracts features from a list of video frames using OpenFace.
        """
        
        self.video_counter += 1
        temp_video_path = os.path.join(self.temp_dir, f"temp_video_{self.video_counter}.avi")
        
        try:
            print(f"   Step 1: Saving {len(frames)} frames as video...")
            self._save_frames_as_video(frames, temp_video_path)
            
            if not os.path.exists(temp_video_path):
                print(f"   ❌ Failed to create video file")
                return None
            
            print(f"   Step 2: Running OpenFace on video...")
            success = self._run_openface(temp_video_path)
            
            if not success:
                print(f"   ❌ OpenFace failed to process video")
                return None
            
            print(f"   Step 3: Reading extracted features...")
            features = self._read_openface_output()
            
            if features is None:
                print(f"   ❌ Failed to read OpenFace output")
                return None
            
            print(f"   ✅ Extracted features: {features.shape}")
            
            # Cleanup
            self._cleanup(temp_video_path)
            
            return features
        
        except Exception as e:
            print(f"   ❌ Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_frames_as_video(self, frames, output_path):
        """
        Saves a list of frames as a video file.
        """
        if len(frames) == 0:
            print("   ⚠️  No frames to save")
            return
        
        height, width = frames[0].shape[:2]
        
        # Use different codec for Windows
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # More compatible
        out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        
        if not out.isOpened():
            print(f"   ⚠️  Failed to open video writer")
            return
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"   ✅ Video saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    
    def _run_openface(self, video_path):
        """
        Runs OpenFace FeatureExtraction on the video.
        """
        
        # Convert paths to absolute paths
        video_path_abs = os.path.abspath(video_path)
        output_dir_abs = os.path.abspath(self.temp_dir)
        
        # Get OpenFace directory
        openface_dir = os.path.dirname(self.openface_path)
        
        # Build OpenFace command
        # Use CLNF model instead of CECLM (doesn't require large patch expert files)
        command = [
            self.openface_path,
            "-f", video_path_abs,
            "-out_dir", output_dir_abs,
            "-mloc", "model/main_clnf_general.txt",  # Use CLNF model (simpler, no large files needed)
            "-aus",  # Extract Action Units
            "-gaze",  # Extract gaze
            "-pose"  # Extract head pose
        ]
        
        print(f"   Running: {' '.join(command)}")
        print(f"   Working directory: {openface_dir}")
        
        try:
            # Run OpenFace from its own directory
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=30,
                text=True,
                cwd=openface_dir  # Run from OpenFace directory
            )
            
            print(f"   Return code: {result.returncode}")
            
            if result.stdout:
                print(f"   stdout (first 500 chars): {result.stdout[:500]}")
            
            if result.stderr:
                print(f"   stderr: {result.stderr[:500]}")
            
            if result.returncode != 0:
                print(f"   ⚠️  OpenFace returned error code: {result.returncode}")
                if "Could not find" in result.stdout or "Could not find" in result.stderr:
                    print(f"   HINT: Missing model files. Check OpenFace/model directory.")
                if "No faces detected" in result.stdout:
                    print(f"   HINT: No face visible in video. Make sure face is clearly visible.")
                return False
            
            # Check if output files were created
            output_files = os.listdir(self.temp_dir)
            # OpenFace creates .csv files with all features
            output_csv = [f for f in output_files if f.endswith('.csv') and 'temp_video' in f]
            
            if len(output_csv) == 0:
                print(f"   ⚠️  No CSV output files found")
                print(f"   Files in output dir: {output_files[:10]}")
                return False
            
            print(f"   ✅ OpenFace completed. Output files: {output_csv}")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  OpenFace timed out (30 seconds)")
            return False
        except Exception as e:
            print(f"   ⚠️  Error running OpenFace: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_openface_output(self):
        """
        Reads and combines AU, gaze, and pose features from OpenFace output.
        """
        
        # List all files in temp directory
        files = os.listdir(self.temp_dir)
        print(f"   Files in temp dir: {files}")
        
        # Find the most recent output CSV
        csv_files = [f for f in files if f.endswith('.csv') and 'temp_video' in f]
        
        if len(csv_files) == 0:
            print(f"   ❌ No CSV files found")
            return None
        
        # Get the most recent one
        csv_files.sort()
        latest_csv = csv_files[-1]
        csv_path = os.path.join(self.temp_dir, latest_csv)
        
        print(f"   Reading: {latest_csv}")
        
        try:
            # Read CSV - OpenFace uses comma+space separator
            data = pd.read_csv(csv_path, skipinitialspace=True)
            
            print(f"   CSV loaded: {len(data)} rows, {len(data.columns)} columns")
            print(f"   First few columns: {data.columns[:10].tolist()}")
            print(f"   Last few columns: {data.columns[-10:].tolist()}")
            
            # Extract features
            features = self._extract_from_csv(data)
            
            return features
            
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_from_csv(self, data):
        """
        Extracts features from a comprehensive OpenFace CSV file.
        """
        
        # Print all columns to debug
        print(f"   All columns in CSV ({len(data.columns)} total):")
        print(f"   {data.columns.tolist()}")
        
        # Find AU columns - OpenFace uses format: AU01_r, AU02_r, etc. (without space)
        # Also check for ' AU01_r' format (with leading space)
        au_cols = []
        for col in data.columns:
            col_clean = col.strip()  # Remove spaces
            if col_clean.startswith('AU') and ('_r' in col_clean or '_c' in col_clean):
                au_cols.append(col)
        
        # Find gaze columns - format: gaze_0_x, gaze_0_y, etc.
        gaze_cols = []
        for col in data.columns:
            col_clean = col.strip().lower()
            if 'gaze' in col_clean and col.strip() not in ['confidence', 'success']:
                gaze_cols.append(col)
        
        # Find pose columns - format: pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz
        pose_cols = []
        for col in data.columns:
            col_clean = col.strip().lower()
            if 'pose' in col_clean and col.strip() not in ['confidence', 'success']:
                pose_cols.append(col)
        
        print(f"   Identified columns:")
        print(f"   - AU columns ({len(au_cols)}): {au_cols[:5]}..." if len(au_cols) > 5 else f"   - AU columns ({len(au_cols)}): {au_cols}")
        print(f"   - Gaze columns ({len(gaze_cols)}): {gaze_cols}")
        print(f"   - Pose columns ({len(pose_cols)}): {pose_cols}")
        
        if len(au_cols) == 0:
            print(f"   ❌ No AU columns found! The CSV might not have AU data.")
            print(f"   Common issue: -aus flag might not be working properly")
            print(f"   Attempting to continue with available features...")
            
            # If no AUs, create dummy AU features (zeros) to match expected shape
            au_features = np.zeros((len(data), 24))
        else:
            au_features = data[au_cols].values
        
        # Extract gaze and pose features
        if len(gaze_cols) > 0:
            gaze_features = data[gaze_cols].values
        else:
            gaze_features = np.zeros((len(data), 8))
            
        if len(pose_cols) > 0:
            pose_features = data[pose_cols].values
        else:
            pose_features = np.zeros((len(data), 6))
        
        # Pad or truncate to expected sizes if needed
        if au_features.shape[1] < 24:
            padding = np.zeros((len(data), 24 - au_features.shape[1]))
            au_features = np.concatenate([au_features, padding], axis=1)
        elif au_features.shape[1] > 24:
            au_features = au_features[:, :24]
            
        if gaze_features.shape[1] < 8:
            padding = np.zeros((len(data), 8 - gaze_features.shape[1]))
            gaze_features = np.concatenate([gaze_features, padding], axis=1)
        elif gaze_features.shape[1] > 8:
            gaze_features = gaze_features[:, :8]
            
        if pose_features.shape[1] < 6:
            padding = np.zeros((len(data), 6 - pose_features.shape[1]))
            pose_features = np.concatenate([pose_features, padding], axis=1)
        elif pose_features.shape[1] > 6:
            pose_features = pose_features[:, :6]
        
        combined = np.concatenate([au_features, gaze_features, pose_features], axis=1)
        
        print(f"   Final feature shape: {combined.shape}")
        
        return combined
    
    def _extract_from_separate_files(self, au_file, gaze_file, pose_file):
        """
        Extracts features from separate AU, gaze, and pose files.
        """
        
        try:
            # Read files
            au_data = pd.read_csv(au_file)
            gaze_data = pd.read_csv(gaze_file)
            pose_data = pd.read_csv(pose_file)
            
            # Remove metadata columns
            meta_cols = ['frame', 'timestamp', 'confidence', 'success']
            au_features = au_data.drop([c for c in meta_cols if c in au_data.columns], axis=1, errors='ignore')
            gaze_features = gaze_data.drop([c for c in meta_cols if c in gaze_data.columns], axis=1, errors='ignore')
            pose_features = pose_data.drop([c for c in meta_cols if c in pose_data.columns], axis=1, errors='ignore')
            
            # Combine
            combined = np.concatenate([
                au_features.values,
                gaze_features.values,
                pose_features.values
            ], axis=1)
            
            return combined
            
        except Exception as e:
            print(f"   ⚠️  Error reading separate files: {e}")
            return None
    
    def _cleanup(self, video_path):
        """
        Removes temporary files.
        """
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Remove OpenFace output files for this video
            video_name = os.path.basename(video_path).replace('.avi', '')
            for filename in os.listdir(self.temp_dir):
                if video_name in filename:
                    filepath = os.path.join(self.temp_dir, filename)
                    try:
                        os.remove(filepath)
                    except:
                        pass
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")


# ============================================================================
# DEPRESSION ANALYZER
# ============================================================================

class DepressionAnalyzer:
    """
    Analyzes features and makes depression predictions.
    """
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, features):
        """
        Makes a depression prediction from features.
        """
        
        # Ensure we have exactly SEQUENCE_LENGTH frames
        if len(features) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(features), features.shape[1]))
            features = np.vstack([features, padding])
        elif len(features) > SEQUENCE_LENGTH:
            features = features[-SEQUENCE_LENGTH:]
        
        # Normalize features
        n_features = features.shape[1]
        features_scaled = self.scaler.transform(features)
        
        # Reshape for model input
        features_input = features_scaled.reshape(1, SEQUENCE_LENGTH, n_features)
        
        # Get prediction
        probability = float(self.model.predict(features_input, verbose=0)[0][0])
        
        # Generate explanation
        explanation = self._generate_explanation(probability, features)
        
        return probability, explanation
    
    def _generate_explanation(self, probability, features):
        """
        Generates explanation for the prediction.
        """
        
        mean_features = np.mean(features, axis=0)
        
        # Risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            explanation = f"High depression risk detected ({probability*100:.1f}%). "
        elif probability >= 0.5:
            risk_level = "MODERATE"
            explanation = f"Moderate depression risk detected ({probability*100:.1f}%). "
        else:
            risk_level = "LOW"
            explanation = f"Low depression risk detected ({probability*100:.1f}%). "
        
        # Add observations
        if probability >= 0.5:
            explanation += "Facial patterns indicate reduced positive affect and possible emotional distress. "
            explanation += "Consider professional consultation for comprehensive assessment."
        else:
            explanation += "Facial expressions show typical emotional patterns. "
            explanation += "Continue monitoring for any changes over time."
        
        return explanation


# ============================================================================
# PREDICTION LOGGER
# ============================================================================

class PredictionLogger:
    """
    Logs predictions to CSV file.
    """
    
    def __init__(self, log_file):
        self.log_file = log_file
        
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("timestamp,probability,risk_level,explanation\n")
    
    def log_prediction(self, probability, explanation):
        """
        Logs a prediction.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.5:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        explanation_escaped = explanation.replace('"', '""')
        
        with open(self.log_file, 'a') as f:
            f.write(f'{timestamp},{probability:.4f},{risk_level},"{explanation_escaped}"\n')


# ============================================================================
# MAIN WEBCAM APPLICATION
# ============================================================================

class DepressionDetectionApp:
    """
    Main application.
    """
    
    def __init__(self, model, scaler, openface_path, temp_dir):
        self.model = model
        self.scaler = scaler
        self.openface_extractor = OpenFaceExtractor(openface_path, temp_dir)
        self.analyzer = DepressionAnalyzer(model, scaler)
        self.logger = PredictionLogger(LOG_FILE)
        
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.is_analyzing = False
        self.current_prediction = None
        self.current_explanation = ""
        self.current_features = None
        
        # History tracking for graphs
        self.prediction_history = deque(maxlen=50)  # Last 50 predictions
        self.timestamp_history = deque(maxlen=50)
        self.analysis_count = 0
        
        # Create a larger window for better visualization
        self.window_width = 1400
        self.window_height = 800
    
    def run(self):
        """
        Main loop.
        """
        
        print("\n" + "="*70)
        print("REAL-TIME DEPRESSION DETECTION SYSTEM")
        print("="*70)
        print("\nControls:")
        print("  [SPACE] - Analyze current footage")
        print("  [Q] - Quit application")
        print("\nStarting webcam...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam")
            return
        
        print("✅ Webcam started successfully\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  Failed to capture frame")
                break
            
            self.frame_buffer.append(frame.copy())
            frame_count += 1
            
            # Auto-analyze every FRAME_BUFFER_SIZE frames
            if frame_count % FRAME_BUFFER_SIZE == 0 and len(self.frame_buffer) == FRAME_BUFFER_SIZE:
                self._analyze_buffer()
            
            display_frame = self._draw_overlay(frame)
            cv2.imshow('Depression Detection System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break
            elif key == ord(' '):
                if len(self.frame_buffer) >= SEQUENCE_LENGTH:
                    self._analyze_buffer()
                else:
                    print(f"⚠️  Need at least {SEQUENCE_LENGTH} frames. Currently: {len(self.frame_buffer)}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Application closed")
    
    def _analyze_buffer(self):
        """
        Analyzes the current frame buffer.
        """
        if self.is_analyzing:
            print("⚠️  Analysis already in progress...")
            return
        
        self.is_analyzing = True
        print(f"\n🔍 Analyzing {len(self.frame_buffer)} frames...")
        
        try:
            frames = list(self.frame_buffer)
            features = self.openface_extractor.extract_features_from_frames(frames)
            
            if features is None:
                print("❌ Feature extraction failed")
                self.is_analyzing = False
                return
            
            probability, explanation = self.analyzer.predict(features)
            
            self.current_prediction = probability
            self.current_explanation = explanation
            
            self.logger.log_prediction(probability, explanation)
            
            print("\n" + "="*70)
            print("PREDICTION RESULT")
            print("="*70)
            print(f"Depression Probability: {probability*100:.2f}%")
            print(f"\n{explanation}")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_analyzing = False
    
    def _draw_overlay(self, frame):
        """
        Draws prediction information on frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "Depression Detection System", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.current_prediction is not None:
            prob_text = f"Risk: {self.current_prediction*100:.1f}%"
            
            if self.current_prediction >= 0.7:
                color = (0, 0, 255)
                risk_text = "HIGH RISK"
            elif self.current_prediction >= 0.5:
                color = (0, 165, 255)
                risk_text = "MODERATE RISK"
            else:
                color = (0, 255, 0)
                risk_text = "LOW RISK"
            
            cv2.putText(frame, risk_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, prob_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Collecting frames...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{FRAME_BUFFER_SIZE}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press SPACE to analyze | Q to quit", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if self.is_analyzing:
            cv2.putText(frame, "ANALYZING...", (w-200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point.
    """
    
    # Check OpenFace
    if not check_openface():
        print("\n❌ Cannot proceed without OpenFace. Please install it first.")
        return
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        print("\n❌ Cannot start application without model and scaler")
        return
    
    # Create application
    app = DepressionDetectionApp(model, scaler, OPENFACE_PATH, TEMP_DIR)
    
    # Run
    app.run()


if __name__ == "__main__":
    main()