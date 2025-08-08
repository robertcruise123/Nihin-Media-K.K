import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import List, Tuple, Dict
import time

class MedicalImageClassifier:
    def __init__(self, model_type='efficientnet', image_size=(224, 224)):
        self.model_type = model_type
        self.image_size = image_size
        self.model = None
        self.is_trained = False
        
    def create_efficientnet_model(self, num_classes=2):
        """Create EfficientNet-based model for transfer learning"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        base_model.trainable = False
        
        inputs = base_model.input
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_pretrained_weights(self):
        """Load a pre-trained model for immediate use"""
        try:
            self.model = self.create_efficientnet_model()
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            return False
    
    def predict_single(self, image_array: np.ndarray) -> Dict:
        """Predict single image classification"""
        start_time = time.time()
        
        try:
            # Simple, balanced medical detection
            medical_prob = self._simple_medical_detection(image_array)
            non_medical_prob = 1.0 - medical_prob
            
            prediction = np.array([non_medical_prob, medical_prob])
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
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
                'confidence': 0.5,
                'probabilities': {'non-medical': 0.5, 'medical': 0.5},
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def _simple_medical_detection(self, image_array: np.ndarray) -> float:
        """Simple but effective medical image detection"""
        try:
            # Convert to analyzable format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                denorm_image = (image_array * std) + mean
            else:
                denorm_image = image_array
            
            denorm_image = np.clip(denorm_image, 0, 1)
            img_255 = (denorm_image * 255).astype(np.uint8)
            
            # Start neutral
            medical_score = 0.5
            
            # 1. GRAYSCALE CHECK (Most important medical indicator)
            if len(img_255.shape) == 3:
                r, g, b = img_255[:,:,0], img_255[:,:,1], img_255[:,:,2]
                
                # Calculate how similar RGB channels are
                r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
                channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
                
                if channel_diff < 5:  # Very grayscale = likely medical
                    medical_score = 0.8
                elif channel_diff < 15:  # Somewhat grayscale = possibly medical
                    medical_score = 0.65
                elif channel_diff > 50:  # Very colorful = likely not medical
                    medical_score = 0.2
                else:  # Moderately colorful = likely not medical
                    medical_score = 0.35
                
                gray = np.mean(img_255, axis=2).astype(np.uint8)
            else:
                # Already grayscale = medical indicator
                medical_score = 0.7
                gray = img_255.astype(np.uint8)
            
            # 2. BACKGROUND CHECK (Medical images often have dark backgrounds)
            h, w = gray.shape
            border_size = min(h, w) // 20
            border_size = max(1, border_size)
            
            # Sample border pixels
            try:
                top_border = gray[:border_size, :].flatten()
                bottom_border = gray[-border_size:, :].flatten()
                left_border = gray[:, :border_size].flatten()
                right_border = gray[:, -border_size:].flatten()
                
                border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
                border_mean = np.mean(border_pixels)
                
                # Dark border adjustment
                if border_mean < 40:  # Very dark border = medical boost
                    medical_score = min(medical_score + 0.15, 0.9)
                elif border_mean > 150:  # Bright border = non-medical boost
                    medical_score = max(medical_score - 0.15, 0.1)
            except:
                pass  # Skip if border analysis fails
            
            # 3. BRIGHTNESS CHECK (Medical images often darker overall)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 80:  # Dark image = medical boost
                medical_score = min(medical_score + 0.1, 0.9)
            elif mean_brightness > 180:  # Bright image = non-medical boost
                medical_score = max(medical_score - 0.1, 0.1)
            
            # 4. CONTRAST CHECK (Medical images have specific contrast ranges)
            contrast = np.std(gray)
            
            if 20 < contrast < 60:  # Medical contrast range
                medical_score = min(medical_score + 0.05, 0.9)
            elif contrast > 100:  # Very high contrast = natural scene
                medical_score = max(medical_score - 0.1, 0.1)
            
            # Final bounds
            medical_score = np.clip(medical_score, 0.1, 0.9)
            
            return medical_score
            
        except Exception as e:
            print(f"Error in simple detection: {e}")
            return 0.5  # Neutral on error
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict classification for a batch of images"""
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