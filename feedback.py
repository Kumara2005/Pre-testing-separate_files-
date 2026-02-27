"""
Feedback generation module that creates detailed, human-like feedback
explaining why marks were awarded or deducted.
Updated to support Subject-Aware technical evaluation.
"""
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from utils import get_api_key


class FeedbackGenerator:
    """Generates detailed feedback for student evaluations."""
    
    def __init__(self, use_ai: bool = False):
        """
        Initialize the feedback generator.
        
        Args:
            use_ai: Whether to use AI (Gemini) for generating feedback
        """
        self.use_ai = use_ai
        self.gemini_model = None
        
        if use_ai:
            try:
                api_key = get_api_key("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini for feedback: {e}")
                self.use_ai = False
    
    def generate_page_feedback(
        self,
        page_score: Dict[str, Any],
        teacher_content: str = "",
        student_content: str = ""
    ) -> str:
        """
        Generate feedback for a single page.
        """
        similarity = page_score.get('similarity_score', 0.0)
        marks_awarded = page_score.get('marks_awarded', 0.0)
        max_marks = page_score.get('max_marks', 0.0)
        page_no = page_score.get('page_no', 0)
        technical_analysis = page_score.get('analysis', "") # Captured from compare.py
        
        # If AI is enabled, generate a personalized summary
        if self.use_ai and self.gemini_model and teacher_content and student_content:
            return self._generate_ai_feedback(
                page_no, similarity, marks_awarded, max_marks,
                teacher_content, student_content
            )
        
        # Fallback to template + the detailed technical analysis from the comparator
        template_feedback = self._generate_template_feedback(
            page_no, similarity, marks_awarded, max_marks
        )
        
        if technical_analysis:
            template_feedback += f"\n**🔍 Technical Breakdown:**\n{technical_analysis}"
            
        return template_feedback
    
    def _generate_template_feedback(
        self,
        page_no: int,
        similarity: float,
        marks_awarded: float,
        max_marks: float
    ) -> str:
        """Generate template-based feedback based on similarity score."""
        feedback = f"**Page {page_no} Summary:**\n"
        
        if similarity >= 85:
            feedback += "✅ **Excellent!** Concepts are clearly understood and well-explained."
        elif similarity >= 65:
            feedback += "👍 **Good attempt.** Core concepts identified, but some descriptive details could be improved."
        elif similarity >= 40:
            feedback += "⚠️ **Satisfactory.** Partial understanding of technical markers; needs more depth."
        else:
            feedback += "❌ **Needs Improvement.** Significant gaps in conceptual accuracy or missing information."
            
        feedback += f"\nMarks: {marks_awarded}/{max_marks} (Match: {similarity:.1f}%)\n"
        return feedback
    
    def _generate_ai_feedback(
        self,
        page_no: int,
        similarity: float,
        marks_awarded: float,
        max_marks: float,
        teacher_content: str,
        student_content: str
    ) -> str:
        """Generate AI-powered detailed feedback focused on conceptual intent."""
        prompt = f"""
        Role: Academic Feedback Assistant.
        Task: Provide encouraging, technical feedback based on the evaluation below.
        
        Page: {page_no}
        Marks: {marks_awarded}/{max_marks}
        Conceptual Match: {similarity}%
        
        Teacher's Key: {teacher_content[:800]}
        Student's Answer: {student_content[:800]}
        
        Instructions:
        1. Mention what technical concepts were correctly identified.
        2. Briefly note what was missing or misunderstood.
        3. Keep it constructive and under 100 words.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return f"**Page {page_no} Analysis:**\n\n{response.text}"
        except Exception as e:
            return self._generate_template_feedback(page_no, similarity, marks_awarded, max_marks)
    
    def generate_overall_feedback(self, evaluation: Dict[str, Any]) -> str:
        """Generate the final overall summary for the student report."""
        percentage = evaluation.get('percentage', 0.0)
        grade = evaluation.get('grade', 'N/A')
        status = evaluation.get('status', 'unknown')
        avg_similarity = evaluation.get('average_similarity', 0.0)
        
        feedback = "\n" + "="*60 + "\n"
        feedback += "OVERALL EVALUATION REPORT\n"
        feedback += "="*60 + "\n\n"
        
        if percentage >= 85:
            feedback += "🌟 **Outstanding!** You have a clear grasp of the technical requirements. Excellent work!\n\n"
        elif percentage >= 60:
            feedback += "✅ **Well Done.** Good technical understanding. Focus on refining your descriptive answers for full marks.\n\n"
        elif percentage >= 40:
            feedback += "👍 **Pass.** You understand the basics, but need to study the technical subdivisions more closely.\n\n"
        else:
            feedback += "❌ **Unsatisfactory.** Significant conceptual gaps found. Please review the key and retry.\n\n"
        
        feedback += f"**Final Grade:** {grade}\n"
        feedback += f"**Status:** {status.upper()}\n"
        feedback += f"**Total Marks:** {evaluation['total_score']}/{evaluation['max_score']}\n"
        feedback += f"**Mean Conceptual Accuracy:** {avg_similarity:.1f}%\n\n"
        
        # Consistency Check
        scores = [p.get('similarity_score', 0) for p in evaluation.get('page_scores', [])]
        if len(scores) > 1:
            if max(scores) - min(scores) < 15:
                feedback += "📝 **Note:** Your performance is consistent throughout the paper.\n"
                
        return feedback
    
    def generate_complete_feedback(
        self,
        evaluation: Dict[str, Any],
        teacher_data: Optional[Dict] = None,
        student_data: Optional[Dict] = None
    ) -> str:
        """Combine page-wise and overall feedback into a single string."""
        report = "\n" + "="*60 + "\n"
        report += "DETAILED MARKING FEEDBACK\n"
        report += "="*60 + "\n\n"
        
        # Iterate through page-wise analysis
        for i, page_score in enumerate(evaluation.get('page_scores', [])):
            t_content = ""
            s_content = ""
            
            if teacher_data and student_data:
                t_pages = teacher_data.get('pages', [])
                s_pages = student_data.get('pages', [])
                if i < len(t_pages) and i < len(s_pages):
                    t_content = t_pages[i].get('content', '')
                    s_content = s_pages[i].get('content', '')
            
            report += self.generate_page_feedback(page_score, t_content, s_content)
            report += "\n" + "-"*60 + "\n"
            
        report += self.generate_overall_feedback(evaluation)
        return report