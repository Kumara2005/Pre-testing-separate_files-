"""
Evaluation module for grading student scripts based on logical question mapping.
Calculates final scores by summing marks awarded for individual questions 
extracted from AI analysis rather than physical page counts.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import re

class Evaluator:
    """Handles grading and question-wise mark accumulation."""
    
    def __init__(self, total_marks: float = 100.0, pass_threshold: float = 40.0):
        """
        Initialize the evaluator.
        """
        self.total_marks = total_marks
        self.pass_threshold = pass_threshold
    
    def parse_marks_from_analysis(self, analysis_text: str) -> Dict[str, float]:
        """
        Extracts SCORE_EARNED and SCORE_TOTAL from the AI's analysis string.
        """
        earned = 0.0
        total = 0.0
        
        # Look for SCORE_EARNED: X patterns in the text
        earned_match = re.search(r"SCORE_EARNED:\s*([\d.]+)", analysis_text)
        total_match = re.search(r"SCORE_TOTAL:\s*([\d.]+)", analysis_text)
        
        if earned_match:
            earned = float(earned_match.group(1))
        if total_match:
            total = float(total_match.group(1))
            
        return {"earned": earned, "total": total}

    def evaluate_comparisons(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate comparisons by summing marks for each question identified.
        Dynamically calculates the total possible marks based on the Answer Key.
        """
        if not comparison_results:
            return self._empty_report()

        page_scores = []
        cumulative_earned = 0.0
        cumulative_possible = 0.0
        total_similarity = 0.0

        for i, comparison in enumerate(comparison_results):
            analysis = comparison.get("analysis", "")
            similarity = comparison.get("similarity", 0.0)
            marks = self.parse_marks_from_analysis(analysis)
            page_info = {
                "page_no": comparison.get("student_page_no", i + 1),
                "similarity_score": round(similarity * 100, 2),
                "marks_awarded": marks["earned"],
                "max_marks": marks["total"],
                "analysis": analysis
            }
            page_scores.append(page_info)
            cumulative_earned += marks["earned"]
            cumulative_possible += marks["total"]
            total_similarity += similarity

        # DYNAMIC TOTAL: Use the sum of questions found in the key as the base
        # This ensures that a 10-question 2-mark paper is graded out of 20, not 100.
        dynamic_max = cumulative_possible if cumulative_possible > 0 else self.total_marks

        # Calculate percentage based on the dynamically found total
        percentage = (cumulative_earned / dynamic_max) * 100 if dynamic_max > 0 else 0
        avg_similarity = (total_similarity / len(comparison_results)) * 100 if comparison_results else 0

        status = "pass" if percentage >= self.pass_threshold else "fail"
        grade = self.calculate_grade(percentage)

        return {
            "total_score": round(cumulative_earned, 2),
            "max_score": round(dynamic_max, 2),
            "percentage": round(percentage, 2),
            "average_similarity": round(avg_similarity, 2),
            "status": status,
            "grade": grade,
            "page_scores": page_scores,
            "total_pages_evaluated": len(comparison_results)
        }
    
    def calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade based on percentage."""
        if percentage >= 90: return "A+"
        elif percentage >= 80: return "A"
        elif percentage >= 70: return "B"
        elif percentage >= 60: return "C"
        elif percentage >= 50: return "D"
        elif percentage >= 40: return "E"
        else: return "F"

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "total_score": 0.0,
            "max_score": self.total_marks,
            "percentage": 0.0,
            "status": "fail",
            "grade": "F",
            "page_scores": [],
            "average_similarity": 0.0,
            "total_pages_evaluated": 0
        }
    
    def generate_evaluation_report(
        self,
        comparison_results: List[Dict[str, Any]],
        student_info: Optional[Dict[str, str]] = None,
        teacher_file: str = "",
        student_file: str = ""
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report including metadata."""
        evaluation = self.evaluate_comparisons(comparison_results)
        
        report = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "teacher_file": teacher_file,
                "student_file": student_file,
                "student_info": student_info or {}
            },
            "evaluation": evaluation
        }
        
        return report

    # Inside evaluation.py

   # Inside evaluation.py

    def get_summary(self, evaluation: Dict[str, Any]) -> str:
        """
        Generates a student-friendly, clean summary without tabular clutter.
        Focuses on question-wise results and specific improvement areas.
        """
        # --- TOP LEVEL SUMMARY ---
        summary = f"## 🎓 Final Result: {evaluation['total_score']} / {evaluation['max_score']}\n"
        summary += f"**Grade**: {evaluation['grade']} | **Status**: {evaluation['status'].upper()}\n\n"
        summary += "---\n"

        # --- QUESTION-WISE BREAKDOWN ---
        summary += "### 📝 Question-Wise Breakdown\n"
        for page in evaluation['page_scores']:
            # Extract question label (e.g., Q1, Q2)
            q_nums = re.findall(r"(Q\d+|Question \d+)", page['analysis'])
            q_label = ", ".join(q_nums) if q_nums else f"Item {page['page_no']}"
            
            # Simple clean result line
            summary += f"✅ **{q_label}**: Awarded {page['marks_awarded']} / {page['max_marks']} marks\n"

        summary += "\n---\n"

        # --- OVERALL IMPROVEMENT SUMMARY ---
        summary += "### 🚀 How to Improve Your Score\n"
        
        # Logic to identify major failing areas
        fail_count = sum(1 for p in evaluation['page_scores'] if p['marks_awarded'] == 0)
        
        if fail_count > 0:
            summary += f"You missed out on marks in {fail_count} key areas. Focus on these improvements:\n"
            summary += "* **Conceptual Accuracy**: For technical subjects, ensure you name the specific 'Factor' or 'Formula' requested by the key (e.g., Learning Rate or Accuracy Formula).\n"
            summary += "* **Avoid Irrelevance**: Do not include information about unrelated topics like Backpropagation or KNN unless specifically asked.\n"
            summary += "* **Diagram Precision**: If a single neuron is asked, do not draw a full multi-layer network.\n"
        else:
            summary += "Great job! To reach an A+, focus on precise technical vocabulary and cleaner sentence structures.\n"
        
        return summary