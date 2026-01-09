"""
OpenFace Feature Extraction Processor
=====================================
Handles OpenFace integration for web application.
"""

import cv2
import numpy as np
import pandas as pd
import subprocess
import os
import time

class OpenFaceProcessor:
    """
    Processes video frames through OpenFace to extract facial features.
    """
    
    def __init__(self, openface_path, temp_dir):
        """
        Initialize OpenFace processor.
        
        Args:
            openface_path: Path to OpenFace FeatureExtraction executable
            temp_dir: Directory for temporary files
        """
        self.openface_path = openface_path
        self.temp_dir = temp_dir
        self.video_counter = 0
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"   OpenFace path: {openface_path}")
        print(f"   Temp directory: {temp_dir}")
    
    def extract_features(self, frames):
        """
        Extract features from a list of frames.
        
        Args:
            frames: List of numpy arrays (OpenCV format BGR images)
        
        Returns:
            numpy array of shape (num_frames, 38) or None if failed
        """
        
        self.video_counter += 1
        temp_video_path = os.path.join(self.temp_dir, f"temp_video_{self.video_counter}.avi")
        
        try:
            # Save frames as video
            self._save_frames_as_video(frames, temp_video_path)
            
            # Run OpenFace
            success = self._run_openface(temp_video_path)
            
            if not success:
                return None
            
            # Read features
            features = self._read_features()
            
            # Cleanup
            self._cleanup(temp_video_path)
            
            return features
            
        except Exception as e:
            print(f"   ❌ Feature extraction error: {e}")
            return None
    
    def _save_frames_as_video(self, frames, output_path, fps=30):
        """Save frames as video file."""
        
        if len(frames) == 0:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def _run_openface(self, video_path):
        """Run OpenFace on video file."""
        
        video_path_abs = os.path.abspath(video_path)
        output_dir_abs = os.path.abspath(self.temp_dir)
        openface_dir = os.path.dirname(self.openface_path)
        
        command = [
            self.openface_path,
            "-f", video_path_abs,
            "-out_dir", output_dir_abs,
            "-mloc", "model/main_clnf_general.txt",
            "-aus",
            "-gaze",
            "-pose"
        ]
        
        try:
            print(f"   ⏱️  Running OpenFace (this may take a few minutes)...")
            start_time = time.time()
            
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=600,  # INCREASED: 10 minutes timeout
                text=True,
                cwd=openface_dir
            )
            
            elapsed = time.time() - start_time
            print(f"   ✅ OpenFace completed in {elapsed:.2f} seconds")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ OpenFace timed out after 600 seconds")
            return False
        except Exception as e:
            print(f"   ❌ OpenFace error: {e}")
            return False
    
    def _read_features(self):
        """Read features from OpenFace output CSV."""
        
        # Find most recent CSV
        csv_files = [f for f in os.listdir(self.temp_dir) 
                     if f.endswith('.csv') and f.startswith('temp_video')]
        
        if not csv_files:
            print(f"   ❌ No CSV files found in {self.temp_dir}")
            return None
        
        csv_files.sort()
        latest_csv = csv_files[-1]
        csv_path = os.path.join(self.temp_dir, latest_csv)
        
        print(f"   📊 Reading features from: {latest_csv}")
        
        try:
            # Read CSV
            data = pd.read_csv(csv_path, skipinitialspace=True)
            print(f"   📈 CSV shape: {data.shape}")
            
            # Extract AU columns
            au_cols = [col for col in data.columns 
                      if col.strip().startswith('AU') and ('_r' in col or '_c' in col)]
            
            # Extract gaze columns
            gaze_cols = [col for col in data.columns 
                        if 'gaze' in col.strip().lower()]
            
            # Extract pose columns
            pose_cols = [col for col in data.columns 
                        if 'pose' in col.strip().lower()]
            
            print(f"   📋 Found {len(au_cols)} AU columns, {len(gaze_cols)} gaze columns, {len(pose_cols)} pose columns")
            
            # Get features
            au_features = data[au_cols].values if au_cols else np.zeros((len(data), 24))
            gaze_features = data[gaze_cols].values if gaze_cols else np.zeros((len(data), 8))
            pose_features = data[pose_cols].values if pose_cols else np.zeros((len(data), 6))
            
            # Ensure correct shapes
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
            
            # Combine
            combined = np.concatenate([au_features, gaze_features, pose_features], axis=1)
            
            print(f"   ✅ Combined features shape: {combined.shape}")
            
            return combined
            
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cleanup(self, video_path):
        """Clean up temporary files."""
        
        try:
            # Remove video
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Remove OpenFace outputs
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