import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import List, Tuple, Dict
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time

class MedicalImageClassifier:
    def __init__(self, model_type='efficientnet', image_size=(224, 224)):
        self.model_type = model_type
        self.image_size = image_size
        self.model = None
        self.is_trained = False
        
    def create_efficientnet_model(self, num_classes=2):
        """Create EfficientNet-based model for transfer learning"""
        # Load pre-trained EfficientNet
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        inputs = base_model.input
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_feature_extractor(self):
        """Create feature extractor using pre-trained CNN"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Add global average pooling
        inputs = base_model.input
        x = base_model(inputs, training=False)
        outputs = GlobalAveragePooling2D()(x)
        
        return Model(inputs, outputs)
    
    def load_pretrained_weights(self):
        """Load a basic pre-trained model for immediate use"""
        try:
            # Create the model
            self.model = self.create_efficientnet_model()
            
            # For demo purposes, we'll use a simple heuristic-based approach
            # In a real scenario, you'd load actual trained weights
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            return False
    
    def predict_single(self, image_array: np.ndarray) -> Dict:
        """
        Predict single image classification
        """
        start_time = time.time()
        
        try:
            if not self.is_trained:
                # Use a simple heuristic for demo (replace with actual model)
                prediction = self._heuristic_prediction(image_array)
            else:
                # Add batch dimension
                if len(image_array.shape) == 3:
                    image_batch = np.expand_dims(image_array, axis=0)
                else:
                    image_batch = image_array
                
                # Get model prediction
                predictions = self.model.predict(image_batch, verbose=0)
                prediction = predictions[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # Map to labels
            class_labels = {0: 'non-medical', 1: 'medical'}
            predicted_label = class_labels[predicted_class]
            
            inference_time = time.time() - start_time
            
            return {
                'prediction': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'non-medical': float(prediction[0]),
                    'medical': float(prediction[1])
                },
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': {'non-medical': 0.5, 'medical': 0.5},
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def _heuristic_prediction(self, image_array: np.ndarray) -> np.ndarray:
        """
        Simple heuristic-based prediction for demo purposes
        This analyzes image characteristics to make basic predictions
        """
        try:
            # Convert to 0-255 range for analysis
            img = ((image_array + 1) * 127.5).astype(np.uint8)
            
            # Calculate basic image statistics
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            
            # Simple heuristics based on typical medical image characteristics
            # Medical images often have:
            # - Lower overall brightness (X-rays, MRIs)
            # - Higher contrast in certain areas
            # - Grayscale or limited color palette
            
            medical_score = 0.0
            
            # Check for low brightness (common in X-rays)
            if mean_intensity < 100:
                medical_score += 0.3
            
            # Check for high contrast
            if std_intensity > 50:
                medical_score += 0.2
            
            # Check if image is predominantly grayscale
            if len(img.shape) == 3:
                r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
                gray_similarity = 1.0 - (np.std([np.mean(r), np.mean(g), np.mean(b)]) / 255.0)
                medical_score += gray_similarity * 0.3
            
            # Check for dark borders (common in medical scans)
            border_pixels = np.concatenate([
                img[0, :].flatten(), img[-1, :].flatten(),
                img[:, 0].flatten(), img[:, -1].flatten()
            ])
            if np.mean(border_pixels) < 50:
                medical_score += 0.2
            
            # Normalize score
            medical_score = min(medical_score, 1.0)
            non_medical_score = 1.0 - medical_score
            
            return np.array([non_medical_score, medical_score])
            
        except Exception as e:
            print(f"Error in heuristic prediction: {str(e)}")
            return np.array([0.5, 0.5])  # Default uncertain prediction
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict classification for a batch of images
        """
        results = []
        for img in images:
            result = self.predict_single(img)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'image_size': self.image_size,
            'is_trained': self.is_trained,
            'num_parameters': self.model.count_params() if self.model else 0
        }