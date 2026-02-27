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
                    # Using the specified stable model Gemini 2.5 Flash
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}")
                self.method = "sentence_transformers"
                self.model = SentenceTransformer(model_name)
    
    def compare_with_gemini(self, teacher_context: str, student_text: str, subject: str, mode: str, threshold: float, retry_count: int = 3) -> Dict[str, Any]:
        """
        AI-driven comparison with Subject-Aware logic, Global Question Mapping, and Partial Credit support.
        teacher_context: The full consolidated teacher key (Master Key).
        student_text: Fragment of student script (e.g., 2 questions).
        """
        if not teacher_context or not student_text or not str(teacher_context).strip() or not str(student_text).strip():
            return {"similarity": 0.0, "analysis": "Empty content provided for evaluation."}
        
        # Adjust AI temperature based on mode (Low temp for logical accuracy in 'Other' subjects)
        target_temp = 0.7 if mode == "language" else 0.2

        # Dynamic Role Definition based on Sidebar Selection 
        if mode == "language":
            role_instruction = (
                "Role: Strict Language Auditor. Evaluate for grammar, specific vocabulary, "
                "phrasing, creativity, and opinion-based depth. Evaluation must be STRICT."
            )
        else:
            role_instruction = (
                "Role: Logical Meaning Evaluator. IGNORE minor OCR, grammar, or spelling errors. "
                "Focus strictly on finding KEYWORDS and LOGICAL MEANING. If the student "
                "expresses the correct concept using relevant keywords, award marks."
            )

        prompt = f"""
{role_instruction}
Subject Type: {subject}
Task: Evaluate the Student Answer fragment against the provided Master Teacher Key.

TEACHER'S MASTER KEY (Context Base):
{teacher_context}

STUDENT'S ANSWER FRAGMENT (Current Page):
{student_text}

CRITICAL EVALUATION RULES:
1. QUESTION-WISE MAPPING: Identify Question Numbers (e.g., Q1, 1., Part A) in the Student's Answer.
2. GLOBAL KEY SEARCH: Search the entire 'TEACHER'S MASTER KEY' to find the matching question regardless of where it appears in the key.
3. KEYWORD REWARD: Especially for 'Other Subjects', award full or partial marks (0.5 or 1.0) if the student uses correct keywords even if the sentence structure is poor. 
4. LENIENCY: For non-language subjects, do not award 0 marks if the logical meaning is clear and matches the key.
5. THRESHOLD: Use a conceptual match threshold of {threshold}.

OUTPUT FORMAT (Strict):
SCORE_EARNED: [Total numeric marks awarded for this fragment]
SCORE_TOTAL: [Total possible marks for these questions according to the Key]
ANALYSIS: [Breakdown: Q1: X/2. Explain why marks were given, highlighting matched keywords or logical accuracy.]
"""
        
        for attempt in range(retry_count):
            try:
                # Essential 12-second delay to stay within Gemini Free Tier limits [cite: 5]
                time.sleep(12) 
                
                response = self.gemini_model.generate_content(
                    prompt, 
                    generation_config={"temperature": target_temp}
                )
                resp_text = response.text
                earned = 0.0
                total = 1.0  
                analysis = ""

                # Parse the structured response logic
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

                ratio = max(0.0, min(1.0, earned / total)) if total > 0 else 0.0
                return {"similarity": ratio, "analysis": analysis if analysis else resp_text}
                
            except Exception as e:
                if "429" in str(e):
                    # Extended back-off if hit by rate limit
                    time.sleep(20)
                if attempt == retry_count - 1:
                    return {"similarity": 0.0, "analysis": f"Error: {str(e)}"}
                time.sleep(5)

    async def compare_page_async(self, master_key_content: str, student_page: Dict, subject: str, mode: str, threshold: float) -> Dict[str, Any]:
        """Orchestrates the comparison using the full Master Key context."""
        loop = asyncio.get_event_loop()
        s_content = student_page.get("content") or ""
        
        result = await loop.run_in_executor(
            self.executor, self.compare_with_gemini, master_key_content, s_content, subject, mode, threshold
        )
        result["student_page_no"] = student_page.get("page_no", 0)
        # Pass extraction confidence to the evaluator for display in app.py 
        result["extraction_confidence"] = student_page.get("extraction_confidence", 0.0)
        return result
    
    async def compare_documents(self, teacher_data: Dict, student_data: Dict, subject: str = "Other Subjects", mode: str = "technical", threshold: float = 0.50, master_key_content: str = "") -> List[Dict[str, Any]]:
        """Processes all student pages against the Global Master Key context."""
        student_pages = student_data.get("pages", [])
        
        tasks = []
        for i in range(len(student_pages)):
            tasks.append(self.compare_page_async(master_key_content, student_pages[i], subject, mode, threshold))
        
        return await asyncio.gather(*tasks)

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)