"""
Feedback generation module that creates detailed, human-like feedback
explaining why marks were awarded or deducted.
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
        
        Args:
            page_score: Dictionary containing page scoring information
            teacher_content: Teacher's answer content (optional)
            student_content: Student's answer content (optional)
            
        Returns:
            Detailed feedback string
        """
        similarity = page_score['similarity_score']
        marks_awarded = page_score['marks_awarded']
        max_marks = page_score['max_marks']
        page_no = page_score['page_no']
        
        if self.use_ai and self.gemini_model and teacher_content and student_content:
            return self._generate_ai_feedback(
                page_no, similarity, marks_awarded, max_marks,
                teacher_content, student_content
            )
        else:
            return self._generate_template_feedback(
                page_no, similarity, marks_awarded, max_marks
            )
    
    def _generate_template_feedback(
        self,
        page_no: int,
        similarity: float,
        marks_awarded: float,
        max_marks: float
    ) -> str:
        """
        Generate template-based feedback.
        
        Args:
            page_no: Page number
            similarity: Similarity percentage
            marks_awarded: Marks awarded
            max_marks: Maximum possible marks
            
        Returns:
            Feedback string
        """
        feedback = f"**Page {page_no} Feedback:**\n\n"
        
        if similarity >= 90:
            feedback += "✅ **Excellent work!** Your answer demonstrates exceptional understanding. "
            feedback += "The content closely matches the expected answer with comprehensive coverage of key concepts. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Strengths:** Strong grasp of concepts, detailed explanations, accurate information."
        
        elif similarity >= 80:
            feedback += "✅ **Very good!** Your answer shows strong understanding with minor areas for improvement. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Strengths:** Good conceptual understanding, mostly accurate content.\n"
            feedback += "**Suggestions:** Consider adding more specific details or examples."
        
        elif similarity >= 70:
            feedback += "👍 **Good attempt.** Your answer captures most key points but could benefit from additional detail. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Strengths:** Basic concepts are understood.\n"
            feedback += "**Areas for improvement:** Provide more thorough explanations and include missing key points."
        
        elif similarity >= 60:
            feedback += "⚠️ **Satisfactory.** Your answer addresses the question but lacks depth or misses some important elements. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Strengths:** Some relevant points identified.\n"
            feedback += "**Areas for improvement:** Expand on core concepts, ensure accuracy, and cover all aspects of the question."
        
        elif similarity >= 50:
            feedback += "⚠️ **Needs improvement.** Your answer shows partial understanding but significant gaps remain. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Areas for improvement:** Review core concepts, provide more detailed explanations, "
            feedback += "and ensure your answer directly addresses the question."
        
        elif similarity >= 40:
            feedback += "❌ **Below expectations.** Your answer demonstrates limited understanding of the topic. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Recommendations:** Revisit the study material, focus on understanding fundamental concepts, "
            feedback += "and practice writing more structured answers."
        
        else:
            feedback += "❌ **Significant gaps.** Your answer shows minimal alignment with the expected response. "
            feedback += f"You earned {marks_awarded}/{max_marks} marks.\n\n"
            feedback += "**Recommendations:** Review the course material thoroughly, seek help to understand core concepts, "
            feedback += "and ensure you understand what the question is asking."
        
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
        """
        Generate AI-powered detailed feedback.
        
        Args:
            page_no: Page number
            similarity: Similarity percentage
            marks_awarded: Marks awarded
            max_marks: Maximum possible marks
            teacher_content: Teacher's answer
            student_content: Student's answer
            
        Returns:
            AI-generated feedback string
        """
        prompt = f"""
        As an educational evaluator, provide constructive feedback for a student's answer.
        
        Expected Answer (Teacher's Key):
        {teacher_content[:1000]}  # Limit to avoid token issues
        
        Student's Answer:
        {student_content[:1000]}
        
        Evaluation Details:
        - Page: {page_no}
        - Semantic Similarity: {similarity}%
        - Marks Awarded: {marks_awarded}/{max_marks}
        
        Please provide:
        1. What the student did well
        2. What was missing or incorrect
        3. Specific suggestions for improvement
        4. Overall assessment
        
        Keep the feedback encouraging, specific, and constructive. Limit to 150 words.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return f"**Page {page_no} Feedback:**\n\n{response.text}"
        except Exception as e:
            print(f"Error generating AI feedback: {e}")
            return self._generate_template_feedback(page_no, similarity, marks_awarded, max_marks)
    
    def generate_overall_feedback(self, evaluation: Dict[str, Any]) -> str:
        """
        Generate overall feedback for the entire evaluation.
        
        Args:
            evaluation: Complete evaluation results
            
        Returns:
            Overall feedback string
        """
        percentage = evaluation.get('percentage', 0.0)
        grade = evaluation.get('grade', 'N/A')
        status = evaluation.get('status', 'unknown')
        avg_similarity = evaluation.get('average_similarity', 0.0)
        
        feedback = "\n" + "="*60 + "\n"
        feedback += "OVERALL FEEDBACK\n"
        feedback += "="*60 + "\n\n"
        
        if percentage >= 85:
            feedback += "🌟 **Outstanding Performance!** You have demonstrated excellent mastery of the material. "
            feedback += "Your answers are comprehensive, well-structured, and accurate. Keep up the excellent work!\n\n"
        
        elif percentage >= 70:
            feedback += "✅ **Good Performance!** You have a solid understanding of most concepts. "
            feedback += "With a bit more attention to detail and depth, you can achieve even better results.\n\n"
        
        elif percentage >= 55:
            feedback += "👍 **Satisfactory Performance.** You grasp the basics but need to work on depth and accuracy. "
            feedback += "Review areas where you lost marks and practice writing more comprehensive answers.\n\n"
        
        elif percentage >= 40:
            feedback += "⚠️ **Marginal Performance.** You're just meeting the minimum requirements. "
            feedback += "Significant improvement is needed. Focus on understanding core concepts more thoroughly.\n\n"
        
        else:
            feedback += "❌ **Needs Significant Improvement.** Your performance indicates gaps in understanding. "
            feedback += "Please seek additional help, review the material carefully, and practice regularly.\n\n"
        
        feedback += f"**Final Grade:** {grade}\n"
        feedback += f"**Status:** {status.upper()}\n"
        feedback += f"**Overall Score:** {evaluation['total_score']}/{evaluation['max_score']}\n"
        feedback += f"**Average Semantic Match:** {avg_similarity}%\n\n"
        
        # Consistency analysis
        scores = [page['similarity_score'] for page in evaluation['page_scores']]
        if len(scores) > 1:
            score_variance = max(scores) - min(scores)
            if score_variance > 30:
                feedback += "**Note:** Your performance varies significantly across pages. "
                feedback += "Try to maintain consistency in your answers.\n"
            elif score_variance < 10:
                feedback += "**Note:** Your performance is consistent across all pages. "
                feedback += "Good job maintaining quality throughout!\n"
        
        return feedback
    
    def generate_complete_feedback(
        self,
        evaluation: Dict[str, Any],
        teacher_data: Optional[Dict] = None,
        student_data: Optional[Dict] = None
    ) -> str:
        """
        Generate complete feedback report including page-wise and overall feedback.
        
        Args:
            evaluation: Evaluation results
            teacher_data: Teacher's extracted data (optional, for AI feedback)
            student_data: Student's extracted data (optional, for AI feedback)
            
        Returns:
            Complete feedback report
        """
        complete_feedback = "\n" + "="*60 + "\n"
        complete_feedback += "DETAILED FEEDBACK REPORT\n"
        complete_feedback += "="*60 + "\n\n"
        
        # Page-wise feedback
        for i, page_score in enumerate(evaluation['page_scores']):
            teacher_content = ""
            student_content = ""
            
            if teacher_data and student_data:
                teacher_pages = teacher_data.get('pages', [])
                student_pages = student_data.get('pages', [])
                if i < len(teacher_pages) and i < len(student_pages):
                    teacher_content = teacher_pages[i].get('content', '')
                    student_content = student_pages[i].get('content', '')
            
            page_feedback = self.generate_page_feedback(
                page_score, teacher_content, student_content
            )
            complete_feedback += page_feedback + "\n\n" + "-"*60 + "\n\n"
        
        # Overall feedback
        complete_feedback += self.generate_overall_feedback(evaluation)
        
        return complete_feedback
