#!/usr/bin/env python3
"""
Command Line Interface for Medical Image Classifier
Usage: python cli_interface.py --input <path/url> --type <image/pdf/url>
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.data.preprocessor import ImagePreprocessor
from src.data.url_extractor import URLImageExtractor
from src.data.pdf_extractor import PDFImageExtractor
from src.models.medical_classifier import MedicalImageClassifier
from src.utils.metrics import ModelMetrics

def main():
    parser = argparse.ArgumentParser(description='Medical Image Classifier CLI')
    parser.add_argument('--input', '-i', required=True, help='Input path or URL')
    parser.add_argument('--type', '-t', choices=['image', 'pdf', 'url'], required=True, help='Input type')
    parser.add_argument('--output', '-o', default='results.json', help='Output file path')
    parser.add_argument('--max-images', '-m', type=int, default=10, help='Max images for URL/PDF extraction')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize components
    classifier = MedicalImageClassifier()
    preprocessor = ImagePreprocessor()
    metrics = ModelMetrics()
    
    if args.verbose:
        print("Loading model...")
    
    classifier.load_pretrained_weights()
    
    results = []
    
    try:
        if args.type == 'image':
            # Single image processing
            if args.verbose:
                print(f"Processing image: {args.input}")
            
            if not os.path.exists(args.input):
                print(f"Error: Image file {args.input} not found")
                sys.exit(1)
            
            processed_img = preprocessor.preprocess_image(args.input)
            if processed_img is not None:
                result = classifier.predict_single(processed_img)
                result['source'] = args.input
                results.append(result)
            else:
                print(f"Error: Could not process image {args.input}")
                sys.exit(1)
        
        elif args.type == 'pdf':
            # PDF processing
            if args.verbose:
                print(f"Extracting images from PDF: {args.input}")
            
            if not os.path.exists(args.input):
                print(f"Error: PDF file {args.input} not found")
                sys.exit(1)
            
            pdf_extractor = PDFImageExtractor()
            extracted_images = pdf_extractor.extract_images_from_pdf(args.input)
            
            if not extracted_images:
                print("No images found in PDF")
                sys.exit(1)
            
            if args.verbose:
                print(f"Found {len(extracted_images)} images in PDF")
            
            for i, img_data in enumerate(extracted_images[:args.max_images]):
                if args.verbose:
                    print(f"Processing image {i+1}/{min(len(extracted_images), args.max_images)}")
                
                processed_img = preprocessor.preprocess_image(img_data['data'])
                if processed_img is not None:
                    result = classifier.predict_single(processed_img)
                    result['source'] = f"{args.input} - Page {img_data['page_number']}, Image {img_data['image_index']}"
                    results.append(result)
        
        elif args.type == 'url':
            # URL processing
            if args.verbose:
                print(f"Extracting images from URL: {args.input}")
            
            url_extractor = URLImageExtractor(max_images=args.max_images)
            extracted_images = url_extractor.extract_images_from_url(args.input)
            
            if not extracted_images:
                print("No images found at URL")
                sys.exit(1)
            
            if args.verbose:
                print(f"Found {len(extracted_images)} images at URL")
            
            for i, img_data in enumerate(extracted_images):
                if args.verbose:
                    print(f"Processing image {i+1}/{len(extracted_images)}")
                
                processed_img = preprocessor.preprocess_image(img_data['data'])
                if processed_img is not None:
                    result = classifier.predict_single(processed_img)
                    result['source'] = img_data['url']
                    results.append(result)
        
        # Calculate metrics
        if results:
            batch_metrics = metrics.calculate_batch_metrics(results)
            
            # Prepare output
            output_data = {
                'input': args.input,
                'input_type': args.type,
                'total_images': len(results),
                'summary': batch_metrics,
                'results': results
            }
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            # Print summary
            print(f"\n📊 Classification Results Summary:")
            print(f"Total images processed: {len(results)}")
            print(f"Medical images: {batch_metrics['predictions_count'].get('medical', 0)}")
            print(f"Non-medical images: {batch_metrics['predictions_count'].get('non-medical', 0)}")
            print(f"Average confidence: {batch_metrics['avg_confidence']:.3f}")
            print(f"Total processing time: {batch_metrics['total_processing_time']:.2f}s")
            print(f"Results saved to: {args.output}")
            
            if args.verbose:
                print("\n📋 Detailed Results:")
                for i, result in enumerate(results):
                    print(f"  Image {i+1}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        else:
            print("No images were successfully processed")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()