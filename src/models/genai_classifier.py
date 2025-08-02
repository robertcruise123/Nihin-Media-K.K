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
        Simulate CLIP-like image-text similarity scoring
        In production, this would use actual CLIP model
        """
        try:
            # Simulate image feature extraction
            image_features = self._extract_visual_features(image_array)
            
            # Simulate text feature extraction
            text_features = self._extract_text_features(text_prompt)
            
            # Simulate cosine similarity
            similarity = np.dot(image_features, text_features) / (
                np.linalg.norm(image_features) * np.linalg.norm(text_features)
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error in CLIP simulation: {str(e)}")
            return 0.0
    
    def _extract_visual_features(self, image_array: np.ndarray) -> np.ndarray:
        """
        Simulate visual feature extraction from image
        In production, this would use CLIP's vision encoder
        """
        # Convert to 0-255 range
        img = ((image_array + 1) * 127.5).astype(np.uint8)
        
        # Extract visual characteristics that differentiate medical/non-medical
        features = []
        
        # Color distribution features
        if len(img.shape) == 3:
            # RGB statistics
            features.extend([
                np.mean(img[:,:,0]),  # Red channel mean
                np.mean(img[:,:,1]),  # Green channel mean  
                np.mean(img[:,:,2]),  # Blue channel mean
                np.std(img[:,:,0]),   # Red channel std
                np.std(img[:,:,1]),   # Green channel std
                np.std(img[:,:,2])    # Blue channel std
            ])
            
            # Grayscale tendency (medical images often grayscale)
            gray_similarity = 1.0 - np.std([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]) / 255.0
            features.append(gray_similarity)
        else:
            # Grayscale image
            features.extend([np.mean(img), np.std(img), 1.0, 0.0, 0.0, 0.0, 1.0])
        
        # Texture and contrast features
        features.extend([
            np.std(img),                    # Overall contrast
            np.mean(img),                   # Overall brightness
            len(np.unique(img)) / 256.0,    # Color diversity
        ])
        
        # Edge and structure features (simplified)
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Simple edge detection simulation
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        features.extend([
            np.mean(edges_h),  # Horizontal edge strength
            np.mean(edges_v),  # Vertical edge strength
        ])
        
        # Spatial distribution features
        center_crop = gray[gray.shape[0]//4:3*gray.shape[0]//4, gray.shape[1]//4:3*gray.shape[1]//4]
        border_region = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        
        features.extend([
            np.mean(center_crop),           # Center region brightness
            np.mean(border_region),         # Border region brightness
            np.std(center_crop),            # Center region contrast
        ])
        
        return np.array(features)
    
    def _extract_text_features(self, text_prompt: str) -> np.ndarray:
        """
        Simulate text feature extraction
        In production, this would use CLIP's text encoder
        """
        # Simulate text embeddings based on medical/anatomical keywords
        medical_keywords = [
            'medical', 'x-ray', 'scan', 'mri', 'ct', 'ultrasound', 'radiological',
            'diagnosis', 'clinical', 'anatomy', 'pathology', 'body', 'bone',
            'tissue', 'organ', 'hospital', 'patient', 'doctor'
        ]
        
        non_medical_keywords = [
            'landscape', 'nature', 'animal', 'building', 'architecture', 'outdoor',
            'scenery', 'wildlife', 'natural', 'environment', 'photography',
            'everyday', 'general', 'object', 'scene'
        ]
        
        # Count keyword occurrences
        text_lower = text_prompt.lower()
        medical_score = sum(1 for word in medical_keywords if word in text_lower)
        non_medical_score = sum(1 for word in non_medical_keywords if word in text_lower)
        
        # Create feature vector
        features = [
            medical_score / len(medical_keywords),
            non_medical_score / len(non_medical_keywords),
            len(text_prompt) / 100.0,  # Text length feature
        ]
        
        # Add some learned patterns simulation
        if 'x-ray' in text_lower or 'scan' in text_lower:
            features.extend([1.0, 0.8, 0.9])
        elif 'landscape' in text_lower or 'nature' in text_lower:
            features.extend([0.1, 0.9, 0.8])
        else:
            features.extend([0.5, 0.5, 0.5])
        
        return np.array(features)
    
    def predict_single(self, image_array: np.ndarray) -> Dict:
        """
        Predict using GenAI zero-shot classification
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
            
            # Aggregate scores
            medical_score = np.mean(medical_similarities)
            non_medical_score = np.mean(non_medical_similarities)
            
            # Normalize scores to probabilities
            total_score = medical_score + non_medical_score
            if total_score > 0:
                medical_prob = medical_score / total_score
                non_medical_prob = non_medical_score / total_score
            else:
                medical_prob = 0.5
                non_medical_prob = 0.5
            
            # Apply softmax-like normalization for better calibration
            exp_med = np.exp(medical_score * 2)  # Scale factor for sensitivity
            exp_non_med = np.exp(non_medical_score * 2)
            total_exp = exp_med + exp_non_med
            
            medical_prob = exp_med / total_exp
            non_medical_prob = exp_non_med / total_exp
            
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