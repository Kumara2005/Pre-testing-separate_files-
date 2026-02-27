"""
Text extraction module optimized for Question-Centric Mapping.
Includes Technical Glossary to fix spelling errors and Confidence Scoring.
"""
import asyncio
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import time

class DocumentExtractor:
    """Handles text extraction from PDF or Images using Gemini 2.5 Flash."""
    
    def __init__(self):
        """Initialize the document extractor with Gemini."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # Using 2.5-flash for high-fidelity vision-to-text extraction
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Semaphore ensures only one request is sent to Google at a time to stay in Free Tier.
        self.rate_limit_lock = asyncio.Semaphore(1)
    
    async def transcribe_page(self, image: Image.Image, page_no: int) -> Dict[str, Any]:
        """Transcribes a single page with Technical Glossary and Confidence scoring."""
        async with self.rate_limit_lock:
            try:
                header_instruction = ""
                if page_no == 1:
                    header_instruction = (
                        "First, look at the top right corner of the page. "
                        "Identify the Student Name and Roll Number written there. "
                        "Format them as METADATA_NAME: [Name] and METADATA_ROLL: [Roll No]. "
                    )
                
                # REQUIREMENT: Technical Glossary to fix spelling errors (e.g., indultive -> inductive)
                glossary = (
                    "TECHNICAL GLOSSARY: Inductive, Gradient, Divergence, Activation, "
                    "Penetration, NetBIOS, Vulnerability, Supervised, Reinforcement, MSE. "
                    "Use this glossary to correct minor OCR/handwriting misspellings."
                )

                # UPDATED PROMPT: Specific for Question-Wise Mapping and Confidence
                prompt = (
                    f"{header_instruction}\n"
                    f"{glossary}\n"
                    "Transcribe the text from this exam paper exactly. "
                    "IMPORTANT: Identify all Question Numbers (e.g., Q1, Q2, Part A) on their own line. "
                    "Include an EXTRACTION_CONFIDENCE score between 0 and 1 based on handwriting legibility. "
                    "Return the output in this strict format:\n"
                    "METADATA_NAME: [Name or 'Unknown']\n"
                    "METADATA_ROLL: [Roll No or 'Unknown']\n"
                    "EXTRACTION_CONFIDENCE: [Numeric Score]\n"
                    "CONTENT: [The full transcribed text with Q numbers on new lines as anchors]"
                )
                
                # Sleep 5 seconds between pages during extraction to avoid 429 Rate Limit
                await asyncio.sleep(5) 
                
                response = await self.model.generate_content_async([prompt, image])
                raw_text = response.text if response.text else ""

                # Default values
                student_name = "Unknown"
                roll_no = "Unknown"
                confidence = 0.0
                clean_content = raw_text

                # Parsing the structured response for Metadata and Confidence
                if "METADATA_NAME:" in raw_text and "CONTENT:" in raw_text:
                    try:
                        name_part = raw_text.split("METADATA_NAME:")[1].split("METADATA_ROLL:")[0].strip()
                        roll_part = raw_text.split("METADATA_ROLL:")[1].split("EXTRACTION_CONFIDENCE:")[0].strip()
                        
                        # Extracting Confidence numeric score
                        conf_match = re.search(r"EXTRACTION_CONFIDENCE:\s*([\d.]+)", raw_text)
                        if conf_match:
                            confidence = float(conf_match.group(1))
                            
                        content_part = raw_text.split("CONTENT:")[1].strip()
                        
                        student_name = name_part if name_part else "Unknown"
                        roll_no = roll_part if roll_part else "Unknown"
                        clean_content = content_part
                    except Exception:
                        pass 

                return {
                    "page_no": page_no, 
                    "content": clean_content, 
                    "student_name": student_name,
                    "roll_no": roll_no,
                    "extraction_confidence": confidence, # NEW FEATURE
                    "raw_text": raw_text 
                }
            except Exception as e:
                print(f"❌ Error on page {page_no}: {str(e)}")
                return {
                    "page_no": page_no, 
                    "content": "", 
                    "student_name": "Unknown", 
                    "roll_no": "Unknown",
                    "extraction_confidence": 0.0
                }
        
    async def extract_from_file(self, file_path: str, source: str) -> Dict[str, Any]:
        """Extract text from a file sequentially (one page at a time)."""
        try:
            print(f"📑 Extracting {source} from: {Path(file_path).name}")
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.pdf']:
                images = convert_from_path(file_path, dpi=200)
            else:
                images = [Image.open(file_path)]
            
            pages_content = []
            final_name = "Unknown"
            final_roll = "Unknown"
            
            for i, img in enumerate(images):
                print(f"   > Processing {source} Page {i+1}/{len(images)}...")
                result = await self.transcribe_page(img, i + 1)
                pages_content.append(result)
                
                if i == 0 and source == "student":
                    final_name = result.get("student_name", "Unknown")
                    final_roll = result.get("roll_no", "Unknown")
            
            return {
                "source": source,
                "total_pages": len(pages_content),
                "pages": pages_content,
                "file_name": Path(file_path).name,
                "student_name": final_name,
                "roll_no": final_roll
            }
        except Exception as e:
            print(f"❌ Extraction Error: {str(e)}")
            return {"source": source, "pages": [], "error": str(e)}

async def extract_documents(teacher_path: str, student_path: str, reference_path: Optional[str] = None) -> Dict[str, Any]:
    """Orchestrator for extracting all documents in the correct order."""
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