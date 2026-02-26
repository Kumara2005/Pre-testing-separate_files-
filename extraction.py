"""
Text extraction module with sequential safe-mode processing for free tier.
Uses Gemini 2.5 Flash with one-at-a-time processing to avoid rate limits.
Includes logic to extract Student Name from the document header (top-right).
"""
import asyncio
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import time

class DocumentExtractor:
    """Handles text extraction from PDF or Images using Gemini 2.5 Flash with sequential processing."""
    
    def __init__(self):
        """Initialize the document extractor with Gemini."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Semaphore ensures only one request is sent to Google at a time.
        self.rate_limit_lock = asyncio.Semaphore(1)
    
    async def transcribe_page(self, image: Image.Image, page_no: int) -> Dict[str, Any]:
        """
        Transcribe a single page with rate limiting and specific name-extraction instructions for page 1.
        """
        async with self.rate_limit_lock:
            try:
                # Page 1 specific instructions to find the handwritten name
                name_instruction = ""
                if page_no == 1:
                    name_instruction = "First, look at the top right corner and identify the Student's Name. "
                
                prompt = (
                    f"{name_instruction}Transcribe the text from this exam paper exactly. "
                    f"Return the output in this specific format:\n"
                    f"STUDENT_NAME: [Extracted Name or 'Unknown']\n"
                    f"CONTENT: [The full transcribed text of the page]"
                )
                
                # Standard delay to avoid 429 errors on free tier
                await asyncio.sleep(1)
                
                response = await self.model.generate_content_async([prompt, image])
                raw_text = response.text if response.text else ""
                
                # Parse the structured response
                student_name = "Unknown"
                clean_content = raw_text
                
                if "STUDENT_NAME:" in raw_text and "CONTENT:" in raw_text:
                    parts = raw_text.split("CONTENT:", 1)
                    name_part = parts[0].replace("STUDENT_NAME:", "").strip()
                    student_name = name_part if name_part else "Unknown"
                    clean_content = parts[1].strip()

                return {
                    "page_no": page_no, 
                    "content": clean_content,
                    "student_name": student_name
                }
            except Exception as e:
                print(f"❌ Error on page {page_no}: {str(e)}")
                return {"page_no": page_no, "content": "", "student_name": "Unknown"}
    
    async def extract_from_file(self, file_path: str, source: str) -> Dict[str, Any]:
        """
        Extract text from a PDF or image file sequentially (one page at a time).
        """
        try:
            print(f"📑 Extracting {source} from: {Path(file_path).name}")
            
            file_ext = Path(file_path).suffix.lower()
            images = []
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                img = Image.open(file_path)
                images = [img]
            elif file_ext == '.pdf':
                print(f"   > Converting PDF to high-resolution images...")
                images = convert_from_path(file_path, dpi=200)
            else:
                raise ValueError(f"Unsupported format: {file_ext}. Use PDF or Images.")
            
            pages_content = []
            final_student_name = "Unknown"
            
            for i, img in enumerate(images):
                print(f"   > Processing {source} Page {i+1}...")
                result = await self.transcribe_page(img, i + 1)
                pages_content.append(result)
                
                # Only capture the name from the first page of the student's script
                if i == 0 and source == "student":
                    final_student_name = result.get("student_name", "Unknown")
            
            print(f"✅ Completed extraction for {source}: {len(images)} pages")
            
            return {
                "source": source,
                "total_pages": len(pages_content),
                "pages": pages_content,
                "file_name": Path(file_path).name,
                "student_name_from_sheet": final_student_name
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