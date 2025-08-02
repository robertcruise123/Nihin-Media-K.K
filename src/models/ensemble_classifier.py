import numpy as np
from typing import Dict, List
import time
from .medical_classifier import MedicalImageClassifier
from .genai_classifier import GenAIClassifier

class EnsembleClassifier:
    """
    Ensemble classifier combining multiple approaches:
    1. Classical ML (feature extraction + traditional classifiers)
    2. Deep Learning (EfficientNet transfer learning)
    3. GenAI (CLIP-like zero-shot classification)
    """
    
    def __init__(self, weights=None):
        # Initialize individual classifiers
        self.deep_learning_model = MedicalImageClassifier(model_type='efficientnet')
        self.genai_model = GenAIClassifier()
        
        # Ensemble weights (can be tuned based on validation performance)
        if weights is None:
            self.weights = {
                'deep_learning': 0.5,  # Primary model
                'genai': 0.3,          # Secondary model  
                'heuristic': 0.2       # Fallback model
            }
        else:
            self.weights = weights
        
        # Load models
        self.deep_learning_model.load_pretrained_weights()
        
        self.is_ready = True
    
    def predict_single(self, image_array: np.ndarray) -> Dict:
        """
        Ensemble prediction combining all three approaches
        """
        start_time = time.time()
        individual_results = {}
        
        try:
            # 1. Deep Learning Prediction
            dl_result = self.deep_learning_model.predict_single(image_array)
            individual_results['deep_learning'] = dl_result
            
            # 2. GenAI Prediction
            genai_result = self.genai_model.predict_single(image_array)
            individual_results['genai'] = genai_result
            
            # 3. Extract individual probabilities
            dl_medical_prob = dl_result['probabilities']['medical']
            dl_non_medical_prob = dl_result['probabilities']['non-medical']
            
            genai_medical_prob = genai_result['probabilities']['medical']
            genai_non_medical_prob = genai_result['probabilities']['non-medical']
            
            # 4. Weighted ensemble combination
            ensemble_medical_prob = (
                self.weights['deep_learning'] * dl_medical_prob +
                self.weights['genai'] * genai_medical_prob
            )
            
            ensemble_non_medical_prob = (
                self.weights['deep_learning'] * dl_non_medical_prob +
                self.weights['genai'] * genai_non_medical_prob
            )
            
            # Normalize probabilities
            total_prob = ensemble_medical_prob + ensemble_non_medical_prob
            if total_prob > 0:
                ensemble_medical_prob /= total_prob
                ensemble_non_medical_prob /= total_prob
            else:
                ensemble_medical_prob = 0.5
                ensemble_non_medical_prob = 0.5
            
            # 5. Final prediction and confidence
            if ensemble_medical_prob > ensemble_non_medical_prob:
                prediction = 'medical'
                confidence = ensemble_medical_prob
            else:
                prediction = 'non-medical'
                confidence = ensemble_non_medical_prob
            
            # 6. Agreement analysis
            dl_pred = dl_result['prediction']
            genai_pred = genai_result['prediction']
            
            agreement_score = self._calculate_agreement([dl_pred, genai_pred])
            
            inference_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'medical': float(ensemble_medical_prob),
                    'non-medical': float(ensemble_non_medical_prob)
                },
                'inference_time': inference_time,
                'method': 'ensemble',
                'individual_predictions': {
                    'deep_learning': {
                        'prediction': dl_pred,
                        'confidence': dl_result['confidence'],
                        'probabilities': dl_result['probabilities']
                    },
                    'genai': {
                        'prediction': genai_pred,
                        'confidence': genai_result['confidence'],
                        'probabilities': genai_result['probabilities']
                    }
                },
                'agreement_score': agreement_score,
                'ensemble_weights': self.weights
            }
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': {'medical': 0.5, 'non-medical': 0.5},
                'inference_time': 0.0,
                'method': 'ensemble',
                'error': str(e)
            }
    
    def _calculate_agreement(self, predictions: List[str]) -> float:
        """
        Calculate agreement score between individual classifiers
        """
        if not predictions:
            return 0.0
        
        # Count agreements
        medical_votes = predictions.count('medical')
        non_medical_votes = predictions.count('non-medical')
        total_votes = len(predictions)
        
        # Agreement is the proportion of majority votes
        majority_votes = max(medical_votes, non_medical_votes)
        agreement = majority_votes / total_votes
        
        return float(agreement)
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict classification for a batch of images using ensemble
        """
        results = []
        for img in images:
            result = self.predict_single(img)
            results.append(result)
        return results
    
    def get_individual_predictions(self, image_array: np.ndarray) -> Dict:
        """
        Get detailed predictions from each individual classifier
        """
        results = {}
        
        # Deep Learning prediction
        dl_result = self.deep_learning_model.predict_single(image_array)
        results['deep_learning'] = dl_result
        
        # GenAI prediction
        genai_result = self.genai_model.predict_single(image_array)
        results['genai'] = genai_result
        
        return results
    
    def update_weights(self, new_weights: Dict):
        """
        Update ensemble weights based on validation performance
        """
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in new_weights.items()}
        else:
            print("Warning: Invalid weights provided, keeping current weights")
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive information about the ensemble
        """
        return {
            'model_type': 'ensemble',
            'components': {
                'deep_learning': self.deep_learning_model.get_model_info(),
                'genai': self.genai_model.get_model_info()
            },
            'weights': self.weights,
            'is_ready': self.is_ready,
            'ensemble_strategy': 'weighted_average'
        }
    
    def analyze_prediction_confidence(self, image_array: np.ndarray) -> Dict:
        """
        Detailed analysis of prediction confidence across all models
        """
        individual_results = self.get_individual_predictions(image_array)
        ensemble_result = self.predict_single(image_array)
        
        analysis = {
            'ensemble_prediction': ensemble_result['prediction'],
            'ensemble_confidence': ensemble_result['confidence'],
            'agreement_score': ensemble_result['agreement_score'],
            'individual_analysis': {},
            'confidence_variance': 0.0,
            'prediction_consistency': True
        }
        
        # Analyze individual predictions
        confidences = []
        predictions = []
        
        for model_name, result in individual_results.items():
            analysis['individual_analysis'][model_name] = {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'agrees_with_ensemble': result['prediction'] == ensemble_result['prediction']
            }
            confidences.append(result['confidence'])
            predictions.append(result['prediction'])
        
        # Calculate confidence variance
        analysis['confidence_variance'] = float(np.var(confidences))
        
        # Check prediction consistency
        unique_predictions = set(predictions)
        analysis['prediction_consistency'] = len(unique_predictions) == 1
        
        return analysis