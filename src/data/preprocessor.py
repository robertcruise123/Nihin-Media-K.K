import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import List, Tuple, Union
import io

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def preprocess_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model inference
        Args:
            image_input: Image file path, bytes, numpy array, or PIL Image
        Returns:
            Preprocessed image array
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Unsupported image input type")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            return image_array
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def preprocess_batch(self, images: List[Union[str, bytes, np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Preprocess a batch of images
        """
        processed_images = []
        for img in images:
            processed = self.preprocess_image(img)
            if processed is not None:
                processed_images.append(processed)
        
        if processed_images:
            return np.array(processed_images)
        return np.array([])
    
    def validate_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> bool:
        """
        Validate if image can be processed
        """
        try:
            processed = self.preprocess_image(image_input)
            return processed is not None
        except:
            return False