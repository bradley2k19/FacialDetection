"""
Depression Model Prediction Logic
=================================
Handles model loading and inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

class DepressionPredictor:
    """
    Handles depression prediction from facial features.
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize predictor with model and scaler.
        
        Args:
            model_path: Path to trained Keras model
            scaler_path: Path to feature scaler pickle file
        """
        print(f"   Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        print(f"   Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.sequence_length = 100
    
    def predict(self, features):
        """
        Make depression prediction from features.
        
        Args:
            features: numpy array of shape (frames, 38)
        
        Returns:
            tuple: (probability, explanation, feature_stats)
        """
        
        # Ensure we have exactly 100 frames
        if len(features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])
        elif len(features) > self.sequence_length:
            features = features[-self.sequence_length:]
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Reshape for model input
        features_input = features_scaled.reshape(1, self.sequence_length, features.shape[1])
        
        # Get prediction
        probability = float(self.model.predict(features_input, verbose=0)[0][0])
        
        # Generate explanation
        explanation = self._generate_explanation(probability, features)
        
        # Calculate feature statistics
        feature_stats = self._calculate_feature_stats(features)
        
        return probability, explanation, feature_stats
    
    def _generate_explanation(self, probability, features):
        """Generate human-readable explanation."""
        
        # Calculate feature statistics
        mean_features = np.mean(features, axis=0)
        au_features = mean_features[:17]
        
        # Build explanation
        if probability >= 0.7:
            explanation = f"High depression risk detected ({probability*100:.1f}%). "
            explanation += "Facial patterns indicate significantly reduced positive affect and possible emotional distress. "
            explanation += "Strong recommendation for professional mental health consultation."
        elif probability >= 0.5:
            explanation = f"Moderate depression risk detected ({probability*100:.1f}%). "
            explanation += "Facial expressions show some indicators of reduced emotional expression. "
            explanation += "Consider professional consultation for comprehensive assessment."
        else:
            explanation = f"Low depression risk detected ({probability*100:.1f}%). "
            explanation += "Facial expressions show typical emotional patterns. "
            explanation += "Continue monitoring for any changes over time."
        
        return explanation
    
    def _calculate_feature_stats(self, features):
        """Calculate statistics about features for visualization."""
        
        # AU features (first 17 are regression AUs)
        au_features = features[:, :17]
        avg_aus = np.mean(au_features, axis=0)
        
        # Get top 5 AUs
        top_indices = np.argsort(avg_aus)[-5:][::-1]
        
        au_names = {
            0: 'AU01 - Inner Brow Raiser',
            1: 'AU02 - Outer Brow Raiser',
            2: 'AU04 - Brow Lowerer',
            3: 'AU05 - Upper Lid Raiser',
            4: 'AU06 - Cheek Raiser',
            5: 'AU07 - Lid Tightener',
            6: 'AU09 - Nose Wrinkler',
            7: 'AU10 - Upper Lip Raiser',
            8: 'AU12 - Lip Corner Puller',
            9: 'AU14 - Dimpler',
            10: 'AU15 - Lip Corner Depressor',
            11: 'AU17 - Chin Raiser',
            12: 'AU20 - Lip Stretcher',
            13: 'AU23 - Lip Tightener',
            14: 'AU25 - Lips Part',
            15: 'AU26 - Jaw Drop',
            16: 'AU45 - Blink'
        }
        
        top_aus = []
        for idx in top_indices:
            if idx < len(au_names):
                top_aus.append({
                    'name': au_names[idx],
                    'intensity': float(avg_aus[idx])
                })
        
        return {
            'top_action_units': top_aus,
            'total_frames': len(features),
            'feature_count': features.shape[1]
        }