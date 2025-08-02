import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import os
import sys

# Add src to path for imports
sys.path.append('src')

from src.data.preprocessor import ImagePreprocessor
from src.data.url_extractor import URLImageExtractor
from src.data.pdf_extractor import PDFImageExtractor
from src.models.medical_classifier import MedicalImageClassifier
from src.utils.metrics import ModelMetrics

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Image Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = MedicalImageClassifier()
    st.session_state.classifier.load_pretrained_weights()

if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = ImagePreprocessor()

if 'metrics' not in st.session_state:
    st.session_state.metrics = ModelMetrics()

if 'results' not in st.session_state:
    st.session_state.results = []

# Main title
st.markdown('<h1 class="main-header">🏥 Medical vs Non-Medical Image Classifier</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
processing_mode = st.sidebar.selectbox(
    "Choose Processing Mode:",
    ["Single Image Upload", "PDF Image Extraction", "URL Image Extraction", "Batch Processing"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
model_info = st.session_state.classifier.get_model_info()
st.sidebar.write(f"**Model Type:** {model_info['model_type']}")
st.sidebar.write(f"**Image Size:** {model_info['image_size']}")
st.sidebar.write(f"**Status:** {'✅ Ready' if model_info['is_trained'] else '❌ Not Ready'}")

# Main content area
if processing_mode == "Single Image Upload":
    st.header("📤 Single Image Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a medical or non-medical image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Classification button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Processing image..."):
                    # Preprocess image
                    processed_img = st.session_state.preprocessor.preprocess_image(image)
                    
                    if processed_img is not None:
                        # Get prediction
                        result = st.session_state.classifier.predict_single(processed_img)
                        
                        # Store result
                        st.session_state.results.append(result)
                        st.session_state.metrics.add_prediction(result)
                        
                        # Display results in the second column
                        with col2:
                            st.subheader("🎯 Classification Results")
                            
                            # Prediction
                            prediction = result['prediction']
                            confidence = result['confidence']
                            
                            if prediction == 'medical':
                                st.success(f"**Prediction:** {prediction.upper()}")
                            else:
                                st.info(f"**Prediction:** {prediction.upper()}")
                            
                            st.metric("Confidence Score", f"{confidence:.3f}")
                            st.metric("Processing Time", f"{result['inference_time']:.3f}s")
                            
                            # Probability breakdown
                            st.subheader("📊 Probability Breakdown")
                            probs = result['probabilities']
                            
                            prob_df = pd.DataFrame({
                                'Class': ['Medical', 'Non-Medical'],
                                'Probability': [probs['medical'], probs['non-medical']]
                            })
                            
                            fig = px.bar(
                                prob_df, 
                                x='Class', 
                                y='Probability',
                                title="Classification Probabilities",
                                color='Class',
                                color_discrete_map={'Medical': '#ff7f0e', 'Non-Medical': '#1f77b4'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Error processing image. Please try with a different image.")

elif processing_mode == "PDF Image Extraction":
    st.header("📄 PDF Image Extraction & Classification")
    
    uploaded_pdf = st.file_uploader(
        "Upload a PDF file",
        type=['pdf'],
        help="Upload a PDF containing medical images"
    )
    
    if uploaded_pdf is not None:
        if st.button("🔍 Extract & Classify Images", type="primary"):
            with st.spinner("Extracting images from PDF..."):
                # Extract images
                pdf_extractor = PDFImageExtractor()
                extracted_images = pdf_extractor.extract_images_from_bytes(
                    uploaded_pdf.read(), 
                    uploaded_pdf.name
                )
                
                if extracted_images:
                    st.success(f"✅ Extracted {len(extracted_images)} images from PDF")
                    
                    # Process each image
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, img_data in enumerate(extracted_images):
                        # Preprocess image
                        processed_img = st.session_state.preprocessor.preprocess_image(img_data['data'])
                        
                        if processed_img is not None:
                            # Get prediction
                            result = st.session_state.classifier.predict_single(processed_img)
                            result['source'] = f"Page {img_data['page_number']}, Image {img_data['image_index']}"
                            result['image_data'] = img_data['data']
                            results.append(result)
                        
                        progress_bar.progress((i + 1) / len(extracted_images))
                    
                    # Display results
                    if results:
                        st.session_state.results.extend(results)
                        display_batch_results(results)
                else:
                    st.warning("No images found in the PDF file.")

elif processing_mode == "URL Image Extraction":
    st.header("🌐 URL Image Extraction & Classification")
    
    url_input = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        help="Enter a URL to extract and classify images from"
    )
    
    max_images = st.slider("Maximum images to extract", 1, 50, 10)
    
    if url_input and st.button("🔍 Extract & Classify Images", type="primary"):
        if url_input.startswith(('http://', 'https://')):
            with st.spinner("Extracting images from URL..."):
                # Extract images
                url_extractor = URLImageExtractor(max_images=max_images)
                extracted_images = url_extractor.extract_images_from_url(url_input)
                
                if extracted_images:
                    st.success(f"✅ Extracted {len(extracted_images)} images from URL")
                    
                    # Process each image
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, img_data in enumerate(extracted_images):
                        # Preprocess image
                        processed_img = st.session_state.preprocessor.preprocess_image(img_data['data'])
                        
                        if processed_img is not None:
                            # Get prediction
                            result = st.session_state.classifier.predict_single(processed_img)
                            result['source'] = img_data['url']
                            result['image_data'] = img_data['data']
                            results.append(result)
                        
                        progress_bar.progress((i + 1) / len(extracted_images))
                    
                    # Display results
                    if results:
                        st.session_state.results.extend(results)
                        display_batch_results(results)
                else:
                    st.warning("No images found at the provided URL.")
        else:
            st.error("Please enter a valid URL starting with http:// or https://")

elif processing_mode == "Batch Processing":
    st.header("📦 Batch Image Processing")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple images for batch classification"
    )
    
    if uploaded_files and st.button("🔍 Classify All Images", type="primary"):
        with st.spinner("Processing images..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Load and preprocess image
                image = Image.open(uploaded_file)
                processed_img = st.session_state.preprocessor.preprocess_image(image)
                
                if processed_img is not None:
                    # Get prediction
                    result = st.session_state.classifier.predict_single(processed_img)
                    result['source'] = uploaded_file.name
                    result['image_data'] = uploaded_file.getvalue()
                    results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display results
            if results:
                st.session_state.results.extend(results)
                display_batch_results(results)

# Function to display batch results
def display_batch_results(results):
    """Display results for batch processing"""
    if not results:
        return
    
    st.subheader("📊 Batch Processing Results")
    
    # Calculate metrics
    metrics = st.session_state.metrics.calculate_batch_metrics(results)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", metrics['total_images'])
    with col2:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.3f}")
    with col3:
        st.metric("Medical Images", metrics['predictions_count'].get('medical', 0))
    with col4:
        st.metric("Processing Time", f"{metrics['total_processing_time']:.2f}s")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary", "🎯 Confidence", "⏱️ Performance", "📋 Details"])
    
    with tab1:
        # Prediction summary
        summary_fig = st.session_state.metrics.create_prediction_summary_plot(results)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with tab2:
        # Confidence distribution
        conf_fig = st.session_state.metrics.create_confidence_distribution_plot(results)
        if conf_fig:
            st.plotly_chart(conf_fig, use_container_width=True)
    
    with tab3:
        # Processing time analysis
        time_fig = st.session_state.metrics.create_processing_time_plot(results)
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
    
    with tab4:
        # Detailed results table
        results_df = st.session_state.metrics.create_detailed_results_table(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )
    
    # Image gallery
    st.subheader("🖼️ Image Gallery with Predictions")
    
    # Display images in grid
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results):
                result = results[idx]
                
                with col:
                    # Display image
                    if 'image_data' in result:
                        img = Image.open(io.BytesIO(result['image_data']))
                        st.image(img, use_column_width=True)
                    
                    # Display prediction info
                    pred = result['prediction']
                    conf = result['confidence']
                    
                    if pred == 'medical':
                        st.success(f"**{pred.upper()}**")
                    else:
                        st.info(f"**{pred.upper()}**")
                    
                    st.write(f"Confidence: {conf:.3f}")
                    if 'source' in result:
                        st.caption(f"Source: {result['source'][:30]}...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 Medical Image Classifier v1.0 | Built with Streamlit & TensorFlow</p>
    <p>Supports PDF extraction, URL scraping, and batch processing</p>
</div>
""", unsafe_allow_html=True)