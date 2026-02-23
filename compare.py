"""
Semantic comparison module for comparing student answers with teacher's key.
Supports both Sentence-Transformers (local) and Gemini API (cloud-based).
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
    """Handles semantic comparison between texts using multiple methods."""
    
    def __init__(self, method: str = "gemini", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic comparator.
        
        Args:
            method: Comparison method ("sentence_transformers" or "gemini")
            model_name: Model name for sentence transformers
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
                # Fallback to sentence transformers
                self.method = "sentence_transformers"
                self.model = SentenceTransformer(model_name)
    
    def compare_with_sentence_transformers(self, text1: str, text2: str) -> float:
        """
        Compare two texts using Sentence Transformers.
        
        Args:
            text1: First text (teacher's answer)
            text2: Second text (student's answer)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Handle None or empty texts
        if text1 is None or text2 is None or not str(text1).strip() or not str(text2).strip():
            return 0.0
        
        # Lazy initialization of sentence transformer model (for fallback scenarios)
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generate embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Convert to percentage (0-1 range)
        return float(max(0.0, min(1.0, similarity)))
    
    def compare_with_gemini(self, text1: str, text2: str, retry_count: int = 3) -> Dict[str, Any]:
        """
        Compare two texts using Gemini API for detailed semantic analysis.
        
        Args:
            text1: First text (teacher's answer)
            text2: Second text (student's answer)
            retry_count: Number of retries for API failures
            
        Returns:
            Dictionary with similarity score and analysis
        """
        # Handle None or empty texts
        if text1 is None or text2 is None or not str(text1).strip() or not str(text2).strip():
            return {"similarity": 0.0, "analysis": "Empty content"}
        
        prompt = f"""
        Compare the following two answers based on CONCEPTUAL ACCURACY, not exact word matching.
        
        Teacher's Expected Answer:
        {text1[:2000]}
        
        Student's Answer:
        {text2[:2000]}
        
        Evaluate based on:
        1. Core concepts and understanding demonstrated
        2. Correctness of key ideas and explanations
        3. Completeness of the answer
        4. Logical reasoning and structure
        
        Note: Different wording is acceptable if the concepts are correct.
        
        Provide:
        SCORE: [0-100 based on conceptual accuracy]
        ANALYSIS: [Brief explanation of what was correct, missing, or incorrect]
        """
        
        for attempt in range(retry_count):
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
                
                # Parse the response
                score = 0.0
                analysis = ""
                
                for line in response_text.split('\n'):
                    if line.startswith('SCORE:'):
                        try:
                            score = float(line.split(':')[1].strip()) / 100.0
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
                
                # Handle rate limiting with exponential backoff
                if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                    wait_time = (2 ** attempt) * 2
                    print(f"Rate limit during comparison, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if attempt == retry_count - 1:
                        # Final fallback to sentence transformers
                        print(f"Error with Gemini API after retries: {e}")
                        similarity = self.compare_with_sentence_transformers(text1, text2)
                        return {
                            "similarity": similarity,
                            "analysis": f"Gemini API rate limited, used fallback. Score: {similarity:.2%}"
                        }
                else:
                    if attempt == retry_count - 1:
                        print(f"Error with Gemini API: {e}")
                        similarity = self.compare_with_sentence_transformers(text1, text2)
                        return {
                            "similarity": similarity,
                            "analysis": f"Gemini API error, used fallback. Score: {similarity:.2%}"
                        }
                    time.sleep(1)
    
    def compare_texts(self, teacher_text: str, student_text: str) -> Dict[str, Any]:
        """
        Compare two texts using the configured method.
        
        Args:
            teacher_text: Teacher's answer
            student_text: Student's answer
            
        Returns:
            Dictionary with similarity score and optional analysis
        """
        # Ensure texts are strings, not None
        teacher_text = str(teacher_text) if teacher_text is not None else ""
        student_text = str(student_text) if student_text is not None else ""
        
        if self.method == "gemini":
            return self.compare_with_gemini(teacher_text, student_text)
        else:
            similarity = self.compare_with_sentence_transformers(teacher_text, student_text)
            return {
                "similarity": similarity,
                "analysis": f"Semantic similarity: {similarity:.2%}"
            }
    
    async def compare_page_async(self, teacher_page: Dict, student_page: Dict) -> Dict[str, Any]:
        """
        Asynchronously compare a single page.
        
        Args:
            teacher_page: Dictionary with teacher's page content
            student_page: Dictionary with student's page content
            
        Returns:
            Comparison result
        """
        loop = asyncio.get_event_loop()
        # Ensure content is never None
        teacher_content = teacher_page.get("content") or ""
        student_content = student_page.get("content") or ""
        
        result = await loop.run_in_executor(
            self.executor,
            self.compare_texts,
            teacher_content,
            student_content
        )
        
        result["teacher_page_no"] = teacher_page.get("page_no", 0)
        result["student_page_no"] = student_page.get("page_no", 0)
        
        return result
    
    async def compare_documents(self, teacher_data: Dict, student_data: Dict) -> List[Dict[str, Any]]:
        """
        Compare all pages from teacher and student documents.
        
        Args:
            teacher_data: Extracted teacher document data
            student_data: Extracted student document data
            
        Returns:
            List of comparison results for each page
        """
        teacher_pages = teacher_data.get("pages", [])
        student_pages = student_data.get("pages", [])
        
        # Match pages (assuming 1-to-1 correspondence)
        min_pages = min(len(teacher_pages), len(student_pages))
        
        tasks = []
        for i in range(min_pages):
            tasks.append(self.compare_page_async(teacher_pages[i], student_pages[i]))
        
        # Process all pages concurrently
        results = await asyncio.gather(*tasks)
        
        return results
    
    def __del__(self):
        """Clean up the executor on deletion."""
        self.executor.shutdown(wait=False)


def compare_documents_sync(teacher_data: Dict, student_data: Dict, method: str = "gemini") -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for comparing documents.
    
    Args:
        teacher_data: Extracted teacher document data
        student_data: Extracted student document data
        method: Comparison method to use (default: "gemini")
        
    Returns:
        List of comparison results
    """
    comparator = SemanticComparator(method=method)
    return asyncio.run(comparator.compare_documents(teacher_data, student_data))
