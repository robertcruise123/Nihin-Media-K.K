import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict
import os

class PDFImageExtractor:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from PDF file
        Returns: List of dictionaries with image data and metadata
        """
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            images_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to RGB if CMYK
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Validate image
                        try:
                            img_pil = Image.open(io.BytesIO(img_data))
                            width, height = img_pil.size
                            
                            # Skip very small images (likely icons or decorative)
                            if width < 100 or height < 100:
                                continue
                            
                            images_data.append({
                                'data': img_data,
                                'page_number': page_num + 1,
                                'image_index': img_index + 1,
                                'width': width,
                                'height': height,
                                'source_file': os.path.basename(pdf_path)
                            })
                            
                        except Exception as e:
                            print(f"Error processing image on page {page_num + 1}: {str(e)}")
                            continue
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            return images_data
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def extract_images_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Dict]:
        """
        Extract images from PDF bytes (for uploaded files)
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        try:
                            img_pil = Image.open(io.BytesIO(img_data))
                            width, height = img_pil.size
                            
                            if width < 100 or height < 100:
                                continue
                            
                            images_data.append({
                                'data': img_data,
                                'page_number': page_num + 1,
                                'image_index': img_index + 1,
                                'width': width,
                                'height': height,
                                'source_file': filename
                            })
                            
                        except Exception as e:
                            print(f"Error processing image on page {page_num + 1}: {str(e)}")
                            continue
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            return images_data
            
        except Exception as e:
            print(f"Error processing PDF bytes: {str(e)}")
            return []