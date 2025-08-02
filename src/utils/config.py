import os
import tensorflow as tf

# Model Configuration
MODEL_CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'num_classes': 2
}

# Class Labels
CLASS_LABELS = {
    0: 'non-medical',
    1: 'medical'
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Image preprocessing settings
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Web scraping settings
REQUEST_TIMEOUT = 10
MAX_IMAGES_PER_URL = 50