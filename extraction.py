"""
Text extraction module with sequential safe-mode processing for free tier.
Uses Gemini 2.5 Flash with one-at-a-time processing to avoid rate limits.
Updated to support optional reference paper extraction for few-shot learning.
"""
import asyncio
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import time

# Use a Semaphore of 1 to ensure only ONE page is sent to Google at a time.
# This is the only way to stay safe on the Free Tier.
rate_limit_lock = asyncio.Semaphore(1)


class DocumentExtractor:
    """Handles text extraction from PDF or Images using Gemini 2.5 Flash with sequential processing."""
    
    def __init__(self):
        """Initialize the document extractor with Gemini."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # We use 'gemini-2.5-flash' as specified
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    async def transcribe_page(self, image: Image.Image, page_no: int) -> Dict[str, Any]:
        """
        Transcribe a single page with rate limiting.
        """
        async with rate_limit_lock:
            try:
                prompt = "Transcribe the text from this exam paper exactly. Return only the text content."
                # Add a small 1-second delay between every single request to breathe
                await asyncio.sleep(1)
                
                response = await self.model.generate_content_async([prompt, image])
                # Ensure content is never None
                content = response.text if response.text else ""
                return {"page_no": page_no, "content": content}
            except Exception as e:
                print(f"❌ Error on page {page_no}: {str(e)}")
                return {"page_no": page_no, "content": ""}
    
    async def extract_from_file(self, file_path: str, source: str) -> Dict[str, Any]:
        """
        Extract text from a PDF or image file sequentially (one page at a time).
        
        Args:
            file_path: Path to the PDF or image file
            source: Source identifier ("teacher", "student", or "reference")
        """
        try:
            print(f"📑 Extracting {source} from: {Path(file_path).name}")
            
            # Detect file type by extension
            file_ext = Path(file_path).suffix.lower()
            images = []
            
            # Handle different file types
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                # It's an image file - load directly
                print(f"   > Detected image file format: {file_ext}")
                img = Image.open(file_path)
                images = [img]
            elif file_ext == '.pdf':
                # It's a PDF - convert to images
                print(f"   > Detected PDF file, converting to images...")
                images = convert_from_path(file_path, dpi=200)
            else:
                raise ValueError(f"Unsupported format: {file_ext}. Use PDF or Images.")
            
            pages_content = []
            
            # We process pages one-by-one (Sequential) to prevent 429 errors
            for i, img in enumerate(images):
                print(f"   > Processing {source} Page {i+1}...")
                result = await self.transcribe_page(img, i + 1)
                pages_content.append(result)
            
            print(f"✅ Completed extraction for {source}: {len(pages_content)} pages")
            
            return {
                "source": source,
                "total_pages": len(pages_content),
                "pages": pages_content,
                "file_name": Path(file_path).name
            }
        except Exception as e:
            print(f"❌ Error extracting from {source}: {str(e)}")
            return {
                "source": source,
                "total_pages": 0,
                "pages": [],
                "file_name": Path(file_path).name,
                "error": str(e)
            }


async def extract_documents(teacher_path: str, student_path: str, reference_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text from teacher, student, and optional reference documents SEQUENTIALLY.
    """
    extractor = DocumentExtractor()
    
    # Process sequentially to avoid rate limits
    teacher_data = await extractor.extract_from_file(teacher_path, "teacher")
    student_data = await extractor.extract_from_file(student_path, "student")
    
    reference_data = None
    if reference_path:
        reference_data = await extractor.extract_from_file(reference_path, "reference")
    
    return {
        "teacher_key": teacher_data,
        "student_script": student_data,
        "reference_paper": reference_data,
        "extraction_status": "completed"
    }


def extract_documents_sync(teacher_file_path: str, student_file_path: str, reference_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper for extracting documents.
    """
    return asyncio.run(extract_documents(teacher_file_path, student_file_path, reference_file_path))