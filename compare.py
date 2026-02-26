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
    def __init__(self, method: str = "gemini", model_name: str = "all-MiniLM-L6-v2"):
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
        if text1 is None or text2 is None or not str(text1).strip() or not str(text2).strip():
            return 0.0
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(max(0.0, min(1.0, similarity)))
    def compare_with_gemini(self, text1: str, text2: str, subject: str = "General", ref_text: Optional[str] = None, retry_count: int = 3) -> Dict[str, Any]:
        if not text1 or not text2 or not str(text1).strip() or not str(text2).strip():
            return {"similarity": 0.0, "analysis": "Empty content provided for evaluation."}
        
        is_creative = any(s in subject.lower() for s in ["english", "tamil", "language", "literature", "arts"])
        target_temp = 0.7 if is_creative else 0.2 # Slightly higher temp for better reasoning

        prompt = f"""
        System Role: Expert Academic Grader for {subject}.
        
        Task: Compare the Student's Answer against the Teacher's Reference Key. 
        Focus on CONCEPTUAL MATCHING and KEYWORDS rather than exact wording.

        SCORING RUBRIC:
        - Part-A (2 Marks): Full marks if the core fact is present.
        - Part-B (16 Marks): Partial marks for identifying reagents, intermediate steps, or partial explanations.
        
        ---
        TEACHER'S REFERENCE KEY:
        {text1}
        
        STUDENT'S ANSWER:
        {text2}
        ---

        INSTRUCTIONS:
        1. Identify the core concepts present in the Teacher's Key.
        2. Check if the Student's Answer captures those concepts, even if using different words or having minor OCR typos.
        3. Be lenient with handwriting transcription errors.
        4. If it is a Chemistry reaction, award marks for mentioning reagents (like CO, HCl, AlCl3) even if the structure isn't fully described.

        OUTPUT FORMAT:
        SCORE_EARNED: [Numeric total value]
        SCORE_TOTAL: [Total possible value for this section]
        ANALYSIS: [Briefly explain what concepts were found and what were missing.]
        """
        
        for attempt in range(retry_count):
            try:
                response = self.gemini_model.generate_content(
                    prompt, 
                    generation_config={"temperature": target_temp}
                )
                resp_text = response.text
                earned = 0.0
                total = 1.0  
                analysis = ""

                for line in resp_text.split('\n'):
                    line = line.strip()
                    if 'SCORE_EARNED:' in line:
                        val = line.split(':')[1].strip()
                        earned = float(''.join(filter(lambda x: x.isdigit() or x == '.', val)))
                    elif 'SCORE_TOTAL:' in line:
                        val = line.split(':')[1].strip()
                        total = float(''.join(filter(lambda x: x.isdigit() or x == '.', val)))
                    elif line.startswith('ANALYSIS:'):
                        analysis = line.split(':', 1)[1].strip()

                # If the AI fails to follow format, ratio defaults to 0.5 as a safety
                ratio = max(0.0, min(1.0, earned / total)) if total > 0 else 0.0
                return {"similarity": ratio, "analysis": analysis if analysis else resp_text}
                
            except Exception as e:
                if attempt == retry_count - 1:
                    return {"similarity": 0.0, "analysis": f"Error: {str(e)}"}
                time.sleep(2) # Increased sleep for rate limits

    async def compare_page_async(self, teacher_page: Dict, student_page: Dict, subject: str = "General", reference_page: Optional[Dict] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        t_content = teacher_page.get("content") or ""
        s_content = student_page.get("content") or ""
        r_content = reference_page.get("content") if reference_page else None
        
        result = await loop.run_in_executor(
            self.executor, self.compare_with_gemini, t_content, s_content, subject, r_content
        )
        result["teacher_page_no"] = teacher_page.get("page_no", 0)
        result["student_page_no"] = student_page.get("page_no", 0)
        return result
    
    async def compare_documents(self, teacher_data: Dict, student_data: Dict, subject: str = "General", reference_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        teacher_pages = teacher_data.get("pages", [])
        student_pages = student_data.get("pages", [])
        ref_pages = reference_data.get("pages", []) if reference_data else []
        
        min_pages = min(len(teacher_pages), len(student_pages))
        tasks = []
        for i in range(min_pages):
            ref_page = ref_pages[i] if i < len(ref_pages) else None
            tasks.append(self.compare_page_async(teacher_pages[i], student_pages[i], subject, ref_page))
        return await asyncio.gather(*tasks)

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)