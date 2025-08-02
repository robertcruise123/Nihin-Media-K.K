import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelMetrics:
    def __init__(self):
        self.predictions = []
        self.processing_times = []
    
    def add_prediction(self, prediction: Dict):
        """Add a prediction result to metrics tracking"""
        self.predictions.append(prediction)
        if 'inference_time' in prediction:
            self.processing_times.append(prediction['inference_time'])
    
    def calculate_batch_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for a batch of predictions"""
        if not results:
            return {}
        
        # Extract metrics
        confidences = [r.get('confidence', 0) for r in results]
        times = [r.get('inference_time', 0) for r in results]
        predictions = [r.get('prediction', 'unknown') for r in results]
        
        # Count predictions
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        
        metrics = {
            'total_images': len(results),
            'avg_confidence': np.mean(confidences),
            'avg_processing_time': np.mean(times),
            'total_processing_time': np.sum(times),
            'predictions_count': pred_counts,
            'medical_percentage': pred_counts.get('medical', 0) / len(results) * 100,
            'high_confidence_predictions': len([c for c in confidences if c > 0.8])
        }
        
        return metrics
    
    def create_confidence_distribution_plot(self, results: List[Dict]):
        """Create confidence distribution plot using Plotly"""
        if not results:
            return None
        
        confidences = [r.get('confidence', 0) for r in results]
        predictions = [r.get('prediction', 'unknown') for r in results]
        
        df = pd.DataFrame({
            'confidence': confidences,
            'prediction': predictions
        })
        
        fig = px.histogram(
            df, 
            x='confidence', 
            color='prediction',
            title='Confidence Score Distribution',
            nbins=20,
            labels={'confidence': 'Confidence Score', 'count': 'Number of Images'}
        )
        
        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Number of Images",
            legend_title="Prediction"
        )
        
        return fig
    
    def create_processing_time_plot(self, results: List[Dict]):
        """Create processing time analysis plot"""
        if not results:
            return None
        
        times = [r.get('inference_time', 0) for r in results]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=times,
            nbinsx=20,
            name='Processing Time Distribution',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Image Processing Time Distribution',
            xaxis_title='Processing Time (seconds)',
            yaxis_title='Number of Images',
            showlegend=False
        )
        
        # Add average line
        avg_time = np.mean(times)
        fig.add_vline(
            x=avg_time, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Avg: {avg_time:.3f}s"
        )
        
        return fig
    
    def create_prediction_summary_plot(self, results: List[Dict]):
        """Create prediction summary pie chart"""
        if not results:
            return None
        
        predictions = [r.get('prediction', 'unknown') for r in results]
        pred_counts = pd.Series(predictions).value_counts()
        
        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title='Classification Results Summary'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        return fig
    
    def create_detailed_results_table(self, results: List[Dict], image_sources: List[str] = None) -> pd.DataFrame:
        """Create detailed results table"""
        if not results:
            return pd.DataFrame()
        
        data = []
        for i, result in enumerate(results):
            row = {
                'Image': f"Image_{i+1}",
                'Prediction': result.get('prediction', 'unknown'),
                'Confidence': f"{result.get('confidence', 0):.3f}",
                'Medical_Prob': f"{result.get('probabilities', {}).get('medical', 0):.3f}",
                'Non_Medical_Prob': f"{result.get('probabilities', {}).get('non-medical', 0):.3f}",
                'Processing_Time': f"{result.get('inference_time', 0):.3f}s"
            }
            
            if image_sources and i < len(image_sources):
                row['Source'] = image_sources[i]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_results(self, results: List[Dict], filename: str = "classification_results.csv"):
        """Export results to CSV"""
        df = self.create_detailed_results_table(results)
        df.to_csv(filename, index=False)
        return filename