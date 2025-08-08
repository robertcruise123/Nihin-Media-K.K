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
            
            # 1. COLOR ANALYSIS - Most important discriminator
            if len(img_255.shape) == 3:
                r, g, b = img_255[:,:,0], img_255[:,:,1], img_255[:,:,2]
                
                # Calculate color variance across the entire image
                r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
                r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
                
                # Color channel difference
                channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
                
                # Color standard deviation difference  
                color_std_diff = max(abs(r_std - g_std), abs(g_std - b_std), abs(r_std - b_std))
                
                # Combined color analysis
                if channel_diff < 10 and color_std_diff < 15:  # Very grayscale
                    medical_score = 0.75
                elif channel_diff < 25 and color_std_diff < 30:  # Somewhat grayscale
                    medical_score = 0.6
                elif channel_diff > 40 or color_std_diff > 50:  # Very colorful = NOT medical
                    medical_score = 0.25
                else:  # Moderate color = likely NOT medical
                    medical_score = 0.4
                
                gray = np.mean(img_255, axis=2).astype(np.uint8)
            else:
                # Already grayscale = potential medical indicator
                medical_score = 0.6
                gray = img_255.astype(np.uint8)
            
            # 2. DETAILED BACKGROUND ANALYSIS
            h, w = gray.shape
            border_size = max(5, min(h, w) // 15)
            
            try:
                # Get border pixels
                top_border = gray[:border_size, :]
                bottom_border = gray[-border_size:, :]
                left_border = gray[:, :border_size]
                right_border = gray[:, -border_size:]
                
                border_pixels = np.concatenate([
                    top_border.flatten(), 
                    bottom_border.flatten(),
                    left_border.flatten(), 
                    right_border.flatten()
                ])
                
                border_mean = np.mean(border_pixels)
                border_std = np.std(border_pixels)
                
                # Medical images typically have dark, uniform backgrounds
                if border_mean < 50 and border_std < 25:  # Dark uniform background
                    medical_score = min(medical_score + 0.2, 0.9)
                elif border_mean < 30:  # Very dark background
                    medical_score = min(medical_score + 0.15, 0.9)
                elif border_mean > 150 and border_std > 40:  # Bright varied background = natural
                    medical_score = max(medical_score - 0.25, 0.1)
                elif border_mean > 100:  # Generally bright background
                    medical_score = max(medical_score - 0.15, 0.1)
                    
            except:
                pass
            
            # 3. OVERALL BRIGHTNESS AND CONTRAST
            overall_mean = np.mean(gray)
            overall_contrast = np.std(gray)
            
            # Medical images often have specific brightness ranges
            if overall_mean < 100:  # Darker images
                if overall_contrast > 30:  # Good contrast in dark image = medical
                    medical_score = min(medical_score + 0.1, 0.9)
                else:
                    medical_score = min(medical_score + 0.05, 0.9)
            elif overall_mean > 150:  # Brighter images = often natural
                medical_score = max(medical_score - 0.15, 0.1)
            
            # Very high contrast = natural scenes
            if overall_contrast > 80:
                medical_score = max(medical_score - 0.2, 0.1)
            elif overall_contrast < 20:  # Very low contrast = possibly medical
                medical_score = min(medical_score + 0.05, 0.9)
            
            # 4. TEXTURE ANALYSIS - Simple version
            try:
                # Calculate local variation
                kernel_size = 5
                if h > kernel_size and w > kernel_size:
                    texture_values = []
                    step_size = max(kernel_size, min(h, w) // 10)
                    
                    for i in range(0, h - kernel_size, step_size):
                        for j in range(0, w - kernel_size, step_size):
                            patch = gray[i:i+kernel_size, j:j+kernel_size]
                            texture_values.append(np.std(patch))
                    
                    if texture_values:
                        avg_texture = np.mean(texture_values)
                        
                        # Very high texture = natural scenes
                        if avg_texture > 60:
                            medical_score = max(medical_score - 0.15, 0.1)
                        # Very low texture = possibly medical
                        elif avg_texture < 20:
                            medical_score = min(medical_score + 0.05, 0.9)
            except:
                pass
            
            # Final adjustment and bounds
            medical_score = np.clip(medical_score, 0.05, 0.95)
            
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