import numpy as np
from PIL import Image
import requests
import io
import base64
from typing import Dict, List
import time

class GenAIClassifier:
    """
    GenAI-based classifier using vision-language models
    Implements CLIP-like zero-shot classification approach
    """
    
    def __init__(self):
        self.model_name = "CLIP-like Vision-Language Model"
        self.is_available = True
        
        # Medical and non-medical text prompts for zero-shot classification
        self.medical_prompts = [
            "a medical X-ray image",
            "a medical scan image showing internal body structures", 
            "a radiological image used for medical diagnosis",
            "a medical imaging scan like MRI, CT, or ultrasound",
            "a clinical photograph for medical documentation",
            "a medical illustration showing anatomy or pathology"
        ]
        
        self.non_medical_prompts = [
            "a natural landscape photograph",
            "a photograph of animals or wildlife",
            "an architectural photograph of buildings",
            "a photograph of everyday objects or scenes",
            "a nature photograph showing outdoor scenery",
            "a general photography image not related to medicine"
        ]
    
    def _simulate_clip_embedding(self, image_array: np.ndarray, text_prompt: str) -> float:
        """
        Simulate CLIP-like image-text similarity scoring with improved accuracy
        """
        try:
            # Simulate image feature extraction
            image_features = self._extract_visual_features(image_array)
            
            # Simulate text feature extraction
            text_features = self._extract_text_features(text_prompt)
            
            # Calculate similarity with better scaling
            dot_product = np.dot(image_features, text_features)
            norm_img = np.linalg.norm(image_features)
            norm_text = np.linalg.norm(text_features)
            
            if norm_img == 0 or norm_text == 0:
                return 0.0
                
            similarity = dot_product / (norm_img * norm_text)
            
            # Apply sigmoid scaling for better distribution
            scaled_similarity = 1 / (1 + np.exp(-similarity * 3))
            
            return float(scaled_similarity)
            
        except Exception as e:
            print(f"Error in CLIP simulation: {str(e)}")
            return 0.5  # Neutral similarity on error
    
    def _extract_visual_features(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive visual features that distinguish medical from non-medical images
        """
        # Convert normalized image to 0-255 range for analysis
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Denormalize from ImageNet preprocessing
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            denorm_image = (image_array * std) + mean
        else:
            denorm_image = image_array
        
        denorm_image = np.clip(denorm_image, 0, 1)
        img_255 = (denorm_image * 255).astype(np.uint8)
        
        features = []
        
        # 1. Color and grayscale analysis
        if len(img_255.shape) == 3:
            r, g, b = img_255[:,:,0], img_255[:,:,1], img_255[:,:,2]
            
            # RGB channel statistics
            features.extend([
                np.mean(r) / 255.0,
                np.mean(g) / 255.0,
                np.mean(b) / 255.0,
                np.std(r) / 255.0,
                np.std(g) / 255.0,
                np.std(b) / 255.0
            ])
            
            # Color variance (medical images often grayscale)
            rgb_means = [np.mean(r), np.mean(g), np.mean(b)]
            color_variance = np.var(rgb_means) / (255.0 * 255.0)
            features.append(color_variance)
            
            # Convert to grayscale for further analysis
            gray = np.mean(img_255, axis=2).astype(np.uint8)
        else:
            # Already grayscale
            gray = img_255.astype(np.uint8)
            # Add dummy RGB features
            features.extend([np.mean(gray)/255.0] * 6)
            features.append(0.0)  # No color variance
        
        # 2. Intensity distribution features
        features.extend([
            np.mean(gray) / 255.0,      # Overall brightness
            np.std(gray) / 255.0,       # Overall contrast
            np.min(gray) / 255.0,       # Darkest pixel
            np.max(gray) / 255.0        # Brightest pixel
        ])
        
        # 3. Histogram analysis
        hist, _ = np.histogram(gray, bins=16, range=(0, 255))
        hist_normalized = hist / np.sum(hist)
        
        # Histogram features
        features.extend([
            np.max(hist_normalized),           # Peak frequency
            np.argmax(hist_normalized) / 15.0, # Peak location (normalized)
            len(hist_normalized[hist_normalized > 0.05]) / 16.0  # Number of significant bins
        ])
        
        # 4. Edge and texture features
        # Horizontal and vertical gradients
        grad_y = np.abs(np.diff(gray.astype(float), axis=0))
        grad_x = np.abs(np.diff(gray.astype(float), axis=1))
        
        features.extend([
            np.mean(grad_y) / 255.0,    # Horizontal edge strength
            np.mean(grad_x) / 255.0,    # Vertical edge strength
            np.std(grad_y) / 255.0,     # Edge variation
            np.std(grad_x) / 255.0
        ])
        
        # 5. Spatial distribution features
        h, w = gray.shape
        
        # Center vs border analysis
        center_h = slice(h//4, 3*h//4)
        center_w = slice(w//4, 3*w//4)
        center_region = gray[center_h, center_w]
        
        # Border regions
        border_regions = [
            gray[0:h//8, :],           # Top border
            gray[-h//8:, :],           # Bottom border  
            gray[:, 0:w//8],           # Left border
            gray[:, -w//8:]            # Right border
        ]
        border_pixels = np.concatenate([region.flatten() for region in border_regions])
        
        features.extend([
            np.mean(center_region) / 255.0,    # Center brightness
            np.mean(border_pixels) / 255.0,    # Border brightness
            np.std(center_region) / 255.0,     # Center contrast
            np.std(border_pixels) / 255.0      # Border contrast
        ])
        
        # 6. Medical-specific pattern detection
        # Dark background detection (common in X-rays, MRIs)
        dark_pixel_ratio = np.sum(gray < 50) / gray.size
        features.append(dark_pixel_ratio)
        
        # High contrast region detection
        high_contrast_ratio = np.sum((grad_y > 30) | (grad_x > 30)) / gray.size
        features.append(high_contrast_ratio)
        
        # Circular/oval structure detection (simplified)
        # This is a basic approximation of detecting round structures
        center_y, center_x = h//2, w//2
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Check for intensity patterns that might indicate circular structures
        max_radius = min(h, w) // 4
        if max_radius > 10:
            ring_mask = (distances > max_radius*0.7) & (distances < max_radius*1.3)
            if np.sum(ring_mask) > 0:
                ring_intensity_var = np.var(gray[ring_mask])
                features.append(ring_intensity_var / (255.0 * 255.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 7. Texture complexity
        # Local standard deviation as texture measure
        texture_complexity = 0.0
        kernel_size = 5
        if h > kernel_size and w > kernel_size:
            for i in range(0, h-kernel_size, kernel_size):
                for j in range(0, w-kernel_size, kernel_size):
                    patch = gray[i:i+kernel_size, j:j+kernel_size]
                    texture_complexity += np.std(patch)
            
            patch_count = ((h-kernel_size)//kernel_size + 1) * ((w-kernel_size)//kernel_size + 1)
            if patch_count > 0:
                texture_complexity = (texture_complexity / patch_count) / 255.0
        
        features.append(texture_complexity)
        
        return np.array(features)
    
    def _extract_text_features(self, text_prompt: str) -> np.ndarray:
        """
        Extract text features that align with visual features for similarity calculation
        """
        # Enhanced medical and non-medical keywords
        medical_keywords = {
            'x-ray': 3.0, 'xray': 3.0, 'scan': 2.5, 'mri': 3.0, 'ct': 2.5,
            'ultrasound': 3.0, 'radiological': 3.0, 'medical': 2.0, 'diagnosis': 2.5,
            'clinical': 2.0, 'anatomy': 2.5, 'pathology': 2.5, 'body': 1.5,
            'bone': 2.0, 'tissue': 2.0, 'organ': 2.0, 'hospital': 1.5,
            'patient': 1.5, 'doctor': 1.5, 'imaging': 2.5, 'radiology': 3.0,
            'internal': 2.0, 'structures': 2.0
        }
        
        non_medical_keywords = {
            'landscape': 2.5, 'nature': 2.0, 'animal': 2.0, 'building': 2.0,
            'architecture': 2.5, 'outdoor': 2.0, 'scenery': 2.0, 'wildlife': 2.5,
            'natural': 2.0, 'environment': 2.0, 'photography': 1.5, 'everyday': 1.5,
            'general': 1.5, 'object': 1.5, 'scene': 1.5, 'street': 2.0,
            'city': 2.0, 'forest': 2.0, 'mountain': 2.0, 'sky': 2.0,
            'water': 2.0, 'tree': 2.0
        }
        
        text_lower = text_prompt.lower()
        
        # Calculate weighted keyword scores
        medical_score = sum(weight for word, weight in medical_keywords.items() if word in text_lower)
        non_medical_score = sum(weight for word, weight in non_medical_keywords.items() if word in text_lower)
        
        # Normalize scores
        total_possible_medical = sum(medical_keywords.values())
        total_possible_non_medical = sum(non_medical_keywords.values())
        
        medical_norm = medical_score / total_possible_medical if total_possible_medical > 0 else 0
        non_medical_norm = non_medical_score / total_possible_non_medical if total_possible_non_medical > 0 else 0
        
        # Create comprehensive feature vector that aligns with visual features
        features = [
            medical_norm,           # Medical keyword strength
            non_medical_norm,       # Non-medical keyword strength
            len(text_prompt) / 100.0,  # Text length
        ]
        
        # Add specific pattern features based on prompt content
        if any(word in text_lower for word in ['x-ray', 'xray', 'scan', 'mri', 'ct']):
            # High medical confidence pattern
            features.extend([0.8, 0.2, 0.9, 0.1, 0.7])  # Medical scan characteristics
        elif any(word in text_lower for word in ['landscape', 'nature', 'outdoor']):
            # High non-medical confidence pattern  
            features.extend([0.1, 0.8, 0.2, 0.9, 0.3])  # Natural scene characteristics
        elif any(word in text_lower for word in ['medical', 'clinical', 'diagnosis']):
            # Medium medical confidence
            features.extend([0.6, 0.4, 0.7, 0.3, 0.5])
        else:
            # Neutral/balanced
            features.extend([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Extend to match visual feature dimensions
        while len(features) < 30:  # Match approximate visual feature count
            features.append(0.5)
        
        return np.array(features[:30])  # Ensure consistent size
    
    def predict_single(self, image_array: np.ndarray) -> Dict:
        """
        Predict using improved GenAI zero-shot classification
        """
        start_time = time.time()
        
        try:
            # Calculate similarities with medical prompts
            medical_similarities = []
            for prompt in self.medical_prompts:
                similarity = self._simulate_clip_embedding(image_array, prompt)
                medical_similarities.append(similarity)
            
            # Calculate similarities with non-medical prompts
            non_medical_similarities = []
            for prompt in self.non_medical_prompts:
                similarity = self._simulate_clip_embedding(image_array, prompt)
                non_medical_similarities.append(similarity)
            
            # Improved scoring with weighted aggregation
            medical_scores = np.array(medical_similarities)
            non_medical_scores = np.array(non_medical_similarities)
            
            # Use weighted mean with higher weight for top similarities
            medical_weights = np.exp(medical_scores * 2) / np.sum(np.exp(medical_scores * 2))
            non_medical_weights = np.exp(non_medical_scores * 2) / np.sum(np.exp(non_medical_scores * 2))
            
            medical_score = np.sum(medical_scores * medical_weights)
            non_medical_score = np.sum(non_medical_scores * non_medical_weights)
            
            # Enhanced normalization with temperature scaling
            temperature = 1.5
            exp_medical = np.exp(medical_score * temperature)
            exp_non_medical = np.exp(non_medical_score * temperature)
            total_exp = exp_medical + exp_non_medical
            
            medical_prob = exp_medical / total_exp
            non_medical_prob = exp_non_medical / total_exp
            
            # Determine prediction
            prediction = "medical" if medical_prob > non_medical_prob else "non-medical"
            confidence = max(medical_prob, non_medical_prob)
            
            inference_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'medical': float(medical_prob),
                    'non-medical': float(non_medical_prob)
                },
                'inference_time': inference_time,
                'method': 'GenAI_CLIP_like',
                'medical_similarities': medical_similarities,
                'non_medical_similarities': non_medical_similarities
            }
            
        except Exception as e:
            print(f"Error in GenAI prediction: {str(e)}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': {'medical': 0.5, 'non-medical': 0.5},
                'inference_time': 0.0,
                'method': 'GenAI_CLIP_like',
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the GenAI model"""
        return {
            'model_type': 'GenAI_CLIP_like',
            'model_name': self.model_name,
            'is_available': self.is_available,
            'zero_shot': True,
            'num_medical_prompts': len(self.medical_prompts),
            'num_non_medical_prompts': len(self.non_medical_prompts)
        }