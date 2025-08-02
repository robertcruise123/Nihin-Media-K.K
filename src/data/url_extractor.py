import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from typing import List, Dict
import time
from PIL import Image
import io

class URLImageExtractor:
    def __init__(self, timeout=10, max_images=50):
        self.timeout = timeout
        self.max_images = max_images
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_images_from_url(self, url: str) -> List[Dict]:
        """
        Extract images from a given URL
        Returns: List of dictionaries with image data and metadata
        """
        try:
            # Get webpage content
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all image tags
            img_tags = soup.find_all('img')
            
            images_data = []
            count = 0
            
            for img_tag in img_tags:
                if count >= self.max_images:
                    break
                
                # Get image source
                img_src = img_tag.get('src') or img_tag.get('data-src')
                if not img_src:
                    continue
                
                # Convert relative URLs to absolute
                img_url = urljoin(url, img_src)
                
                # Download and validate image
                img_data = self._download_image(img_url)
                if img_data:
                    images_data.append({
                        'url': img_url,
                        'data': img_data,
                        'alt_text': img_tag.get('alt', ''),
                        'title': img_tag.get('title', ''),
                        'source_url': url
                    })
                    count += 1
                
                # Rate limiting
                time.sleep(0.1)
            
            return images_data
            
        except Exception as e:
            print(f"Error extracting images from {url}: {str(e)}")
            return []
    
    def _download_image(self, img_url: str) -> bytes:
        """
        Download and validate image from URL
        """
        try:
            response = self.session.get(img_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return None
            
            # Download image data
            img_data = response.content
            
            # Validate image by trying to open it
            try:
                img = Image.open(io.BytesIO(img_data))
                img.verify()  # Verify it's a valid image
                return img_data
            except:
                return None
                
        except Exception as e:
            print(f"Error downloading image {img_url}: {str(e)}")
            return None
    
    def extract_from_multiple_urls(self, urls: List[str]) -> Dict[str, List[Dict]]:
        """
        Extract images from multiple URLs
        """
        results = {}
        for url in urls:
            print(f"Extracting images from: {url}")
            results[url] = self.extract_images_from_url(url)
        return results