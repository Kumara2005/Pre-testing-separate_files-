"""
Evaluation module for grading student scripts based on semantic comparison.
Generates structured scores and stores results in JSON format.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime


class Evaluator:
    """Handles grading and evaluation of student scripts."""
    
    def __init__(self, total_marks: float = 100.0, pass_threshold: float = 40.0):
        """
        Initialize the evaluator.
        
        Args:
            total_marks: Total marks for the assessment
            pass_threshold: Minimum percentage to pass
        """
        self.total_marks = total_marks
        self.pass_threshold = pass_threshold
    
    def calculate_page_score(self, similarity: float, max_marks_per_page: float) -> float:
        """
        Calculate score for a single page based on similarity.
        
        Args:
            similarity: Similarity score (0-1)
            max_marks_per_page: Maximum marks allocated for this page
            
        Returns:
            Calculated score for the page
        """
        # Apply a scoring curve (can be customized)
        if similarity >= 0.9:
            score_multiplier = 1.0
        elif similarity >= 0.8:
            score_multiplier = 0.95
        elif similarity >= 0.7:
            score_multiplier = 0.85
        elif similarity >= 0.6:
            score_multiplier = 0.75
        elif similarity >= 0.5:
            score_multiplier = 0.65
        elif similarity >= 0.4:
            score_multiplier = 0.50
        elif similarity >= 0.3:
            score_multiplier = 0.35
        else:
            score_multiplier = similarity  # Linear below 30%
        
        return round(max_marks_per_page * score_multiplier, 2)
    
    def evaluate_comparisons(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate all page comparisons and calculate overall score.
        
        Args:
            comparison_results: List of comparison results from compare module
            
        Returns:
            Comprehensive evaluation results
        """
        if not comparison_results:
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
        
        # Calculate marks per page
        marks_per_page = self.total_marks / len(comparison_results)
        
        page_scores = []
        total_similarity = 0.0
        total_score = 0.0
        
        for i, comparison in enumerate(comparison_results):
            similarity = comparison.get("similarity", 0.0)
            page_score = self.calculate_page_score(similarity, marks_per_page)
            
            page_info = {
                "page_no": comparison.get("student_page_no", i + 1),
                "similarity_score": round(similarity * 100, 2),
                "marks_awarded": page_score,
                "max_marks": round(marks_per_page, 2),
                "analysis": comparison.get("analysis", "")
            }
            
            page_scores.append(page_info)
            total_similarity += similarity
            total_score += page_score
        
        # Calculate overall metrics
        average_similarity = total_similarity / len(comparison_results)
        percentage = (total_score / self.total_marks) * 100
        status = "pass" if percentage >= self.pass_threshold else "fail"
        
        # Determine grade
        grade = self.calculate_grade(percentage)
        
        return {
            "total_score": round(total_score, 2),
            "max_score": self.total_marks,
            "percentage": round(percentage, 2),
            "average_similarity": round(average_similarity * 100, 2),
            "status": status,
            "grade": grade,
            "page_scores": page_scores,
            "total_pages_evaluated": len(comparison_results)
        }
    
    def calculate_grade(self, percentage: float) -> str:
        """
        Calculate letter grade based on percentage.
        
        Args:
            percentage: Score percentage
            
        Returns:
            Letter grade
        """
        if percentage >= 90:
            return "A+"
        elif percentage >= 85:
            return "A"
        elif percentage >= 80:
            return "A-"
        elif percentage >= 75:
            return "B+"
        elif percentage >= 70:
            return "B"
        elif percentage >= 65:
            return "B-"
        elif percentage >= 60:
            return "C+"
        elif percentage >= 55:
            return "C"
        elif percentage >= 50:
            return "C-"
        elif percentage >= 45:
            return "D+"
        elif percentage >= 40:
            return "D"
        else:
            return "F"
    
    def generate_evaluation_report(
        self,
        comparison_results: List[Dict[str, Any]],
        student_info: Optional[Dict[str, str]] = None,
        teacher_file: str = "",
        student_file: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            comparison_results: List of comparison results
            student_info: Optional dictionary with student information
            teacher_file: Name of teacher's answer key file
            student_file: Name of student's script file
            
        Returns:
            Complete evaluation report
        """
        evaluation = self.evaluate_comparisons(comparison_results)
        
        report = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "teacher_file": teacher_file,
                "student_file": student_file,
                "student_info": student_info or {}
            },
            "evaluation": evaluation,
            "detailed_page_analysis": evaluation["page_scores"]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation report to a JSON file.
        
        Args:
            report: Evaluation report dictionary
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def load_report(self, report_path: str) -> Dict[str, Any]:
        """
        Load evaluation report from a JSON file.
        
        Args:
            report_path: Path to the report file
            
        Returns:
            Evaluation report dictionary
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_summary(self, evaluation: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the evaluation.
        
        Args:
            evaluation: Evaluation results dictionary
            
        Returns:
            Formatted summary string
        """
        summary = f"""
EVALUATION SUMMARY
==================
Total Score: {evaluation['total_score']}/{evaluation['max_score']}
Percentage: {evaluation['percentage']}%
Grade: {evaluation['grade']}
Status: {evaluation['status'].upper()}
Average Similarity: {evaluation['average_similarity']}%
Pages Evaluated: {evaluation['total_pages_evaluated']}

Performance Breakdown:
"""
        
        for page in evaluation['page_scores']:
            summary += f"\nPage {page['page_no']}: {page['marks_awarded']}/{page['max_marks']} marks "
            summary += f"(Similarity: {page['similarity_score']}%)"
        
        return summary
