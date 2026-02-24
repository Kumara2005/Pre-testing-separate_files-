"""
Semantic comparison module for comparing student answers with teacher's key.
Supports both Sentence-Transformers (local) and Gemini API (cloud-based).
Updated to support few-shot learning using an optional teacher-corrected reference.
"""
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from utils import get_api_key
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


class SemanticComparator:
    """Handles semantic comparison between texts using multiple methods and few-shot calibration."""
    
    def __init__(self, method: str = "gemini", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic comparator.
        """
        self.method = method
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        if method == "sentence_transformers":
            self.model = SentenceTransformer(model_name)
        elif method == "gemini":
            try:
                api_key = get_api_key("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}")
                self.method = "sentence_transformers"
                self.model = SentenceTransformer(model_name)
    
    def compare_with_sentence_transformers(self, text1: str, text2: str) -> float:
        """Compare two texts using Sentence Transformers (Baseline)."""
        if text1 is None or text2 is None or not str(text1).strip() or not str(text2).strip():
            return 0.0
        
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(max(0.0, min(1.0, similarity)))
    
    def compare_with_gemini(self, text1: str, text2: str, ref_text: Optional[str] = None, retry_count: int = 3) -> Dict[str, Any]:
        """
        Compare student answer with teacher's key using Few-Shot Learning.
        
        Args:
            text1: Teacher's Expected Answer
            text2: Student's Answer
            ref_text: Optional human-corrected reference (Few-Shot data)
        """
        if text1 is None or text2 is None or not str(text1).strip() or not str(text2).strip():
            return {"similarity": 0.0, "analysis": "Empty content"}
        
        # Build Calibration Context for Few-Shot Learning
        calibration_context = ""
        if ref_text and str(ref_text).strip():
            calibration_context = f"""
            CALIBRATION REFERENCE (Few-Shot Example):
            Use the following human-corrected sample to understand the grading standards, 
            leniency, and feedback style preferred by the teacher:
            ---
            {ref_text[:1500]}
            ---
            """

        prompt = f"""
        Compare the student's answer against the teacher's expected answer based on CONCEPTUAL ACCURACY.
        {calibration_context}
        
        Teacher's Expected Answer:
        {text1[:2000]}
        
        Student's Answer:
        {text2[:2000]}
        
        Instructions:
        1. Evaluate core concepts, correctness, and logical structure.
        2. If a CALIBRATION REFERENCE is provided, align your scoring strictness with it.
        3. Different wording is acceptable if concepts are correct.
        
        Provide the output in exactly this format:
        SCORE: [0-100]
        ANALYSIS: [Brief explanation of correctness and alignment with reference if used]
        """
        
        for attempt in range(retry_count):
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
                
                score = 0.0
                analysis = ""
                
                for line in response_text.split('\n'):
                    if line.startswith('SCORE:'):
                        try:
                            score = float(line.split(':')[1].strip().replace('%', '')) / 100.0
                        except:
                            score = 0.5
                    elif line.startswith('ANALYSIS:'):
                        analysis = line.split(':', 1)[1].strip()
                
                return {
                    "similarity": max(0.0, min(1.0, score)),
                    "analysis": analysis if analysis else response_text
                }
            
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    if attempt == retry_count - 1:
                        similarity = self.compare_with_sentence_transformers(text1, text2)
                        return {"similarity": similarity, "analysis": "Fallback used due to rate limit."}
                else:
                    if attempt == retry_count - 1:
                        similarity = self.compare_with_sentence_transformers(text1, text2)
                        return {"similarity": similarity, "analysis": f"Error: {str(e)}"}
                    time.sleep(1)
    
    def compare_texts(self, teacher_text: str, student_text: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """Entry point for comparison logic."""
        if self.method == "gemini":
            return self.compare_with_gemini(teacher_text, student_text, reference_text)
        else:
            similarity = self.compare_with_sentence_transformers(teacher_text, student_text)
            return {"similarity": similarity, "analysis": f"Semantic match: {similarity:.2%}"}
    
    async def compare_page_async(self, teacher_page: Dict, student_page: Dict, reference_page: Optional[Dict] = None) -> Dict[str, Any]:
        """Asynchronously compare a single page with optional few-shot calibration."""
        loop = asyncio.get_event_loop()
        t_content = teacher_page.get("content") or ""
        s_content = student_page.get("content") or ""
        r_content = reference_page.get("content") if reference_page else None
        
        result = await loop.run_in_executor(
            self.executor,
            self.compare_texts,
            t_content,
            s_content,
            r_content
        )
        
        result["teacher_page_no"] = teacher_page.get("page_no", 0)
        result["student_page_no"] = student_page.get("page_no", 0)
        return result
    
    async def compare_documents(self, teacher_data: Dict, student_data: Dict, reference_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Compare documents page-by-page, incorporating few-shot reference if available."""
        teacher_pages = teacher_data.get("pages", [])
        student_pages = student_data.get("pages", [])
        ref_pages = reference_data.get("pages", []) if reference_data else []
        
        min_pages = min(len(teacher_pages), len(student_pages))
        tasks = []
        for i in range(min_pages):
            ref_page = ref_pages[i] if i < len(ref_pages) else None
            tasks.append(self.compare_page_async(teacher_pages[i], student_pages[i], ref_page))
        
        return await asyncio.gather(*tasks)
    
    def __del__(self):
        self.executor.shutdown(wait=False)


def compare_documents_sync(teacher_data: Dict, student_data: Dict, reference_data: Optional[Dict] = None, method: str = "gemini") -> List[Dict[str, Any]]:
    """Synchronous wrapper for document comparison."""
    comparator = SemanticComparator(method=method)
    return asyncio.run(comparator.compare_documents(teacher_data, student_data, reference_data))