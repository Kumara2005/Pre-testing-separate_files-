"""
Pipeline orchestrator that connects all modules into a seamless workflow.
Coordinates extraction, comparison, evaluation, and feedback generation.
"""
import asyncio
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time

from extraction import DocumentExtractor, extract_documents
from compare import SemanticComparator
from evaluation import Evaluator
from feedback import FeedbackGenerator
from utils import save_json, ensure_directory_exists, verify_gemini_api_key, check_api_prerequisites


class CorrectionPipeline:
    """Orchestrates the complete paper correction workflow."""
    
    def __init__(
        self,
        comparison_method: str = "gemini",
        use_ai_feedback: bool = False,
        total_marks: float = 100.0,
        output_dir: str = "results"
    ):
        """
        Initialize the correction pipeline.
        
        Args:
            comparison_method: Method for semantic comparison (default: "gemini")
            use_ai_feedback: Whether to use AI for feedback generation
            total_marks: Total marks for the assessment
            output_dir: Directory to save results
        """
        self.comparison_method = comparison_method
        self.use_ai_feedback = use_ai_feedback
        self.total_marks = total_marks
        self.output_dir = output_dir
        
        # Initialize components
        self.extractor = DocumentExtractor()
        self.comparator = SemanticComparator(method=comparison_method)
        self.evaluator = Evaluator(total_marks=total_marks)
        self.feedback_generator = FeedbackGenerator(use_ai=use_ai_feedback)
        
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
    
    async def extract_phase(
        self,
        teacher_file_path: str,
        student_file_path: str
    ) -> Dict[str, Any]:
        """
        Phase 1: Extract text from both documents using sequential safe-mode.
        
        Args:
            teacher_file_path: Path to teacher's answer key
            student_file_path: Path to student's script
            
        Returns:
            Extracted data from both documents
        """
        print("📄 Phase 1: Sequential Extraction (Free Tier Safe Mode)...")
        start_time = time.time()
        
        extracted_data = await extract_documents(teacher_file_path, student_file_path)
        
        # Validate that we actually got text
        t_text = sum(len(p.get('content', '')) for p in extracted_data['teacher_key']['pages'])
        s_text = sum(len(p.get('content', '')) for p in extracted_data['student_script']['pages'])
        
        if t_text == 0 or s_text == 0:
            print("⚠️ WARNING: Extraction returned empty text. Check API Key/Connection.")
            print(f"   Teacher content length: {t_text} chars")
            print(f"   Student content length: {s_text} chars")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Extraction completed in {elapsed_time:.2f} seconds")
        print(f"   Teacher pages: {extracted_data['teacher_key']['total_pages']} ({t_text} chars)")
        print(f"   Student pages: {extracted_data['student_script']['total_pages']} ({s_text} chars)")
        
        return extracted_data
    
    async def comparison_phase(self, extracted_data: Dict[str, Any]) -> list:
        """
        Phase 2: Compare student script with teacher's key.
        
        Args:
            extracted_data: Data from extraction phase
            
        Returns:
            List of comparison results
        """
        print("\n🔍 Phase 2: Comparing student answers with teacher's key...")
        start_time = time.time()
        
        teacher_data = extracted_data['teacher_key']
        student_data = extracted_data['student_script']
        
        comparison_results = await self.comparator.compare_documents(
            teacher_data, student_data
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Comparison completed in {elapsed_time:.2f} seconds")
        print(f"   Pages compared: {len(comparison_results)}")
        
        return comparison_results
    
    def evaluation_phase(self, comparison_results: list, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Evaluate and generate scores.
        
        Args:
            comparison_results: Results from comparison phase
            extracted_data: Original extracted data
            
        Returns:
            Complete evaluation report
        """
        print("\n📊 Phase 3: Evaluating and generating scores...")
        start_time = time.time()
        
        teacher_file = extracted_data['teacher_key'].get('file_name', 'teacher_key.pdf')
        student_file = extracted_data['student_script'].get('file_name', 'student_script.pdf')
        
        evaluation_report = self.evaluator.generate_evaluation_report(
            comparison_results=comparison_results,
            teacher_file=teacher_file,
            student_file=student_file
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Evaluation completed in {elapsed_time:.2f} seconds")
        print(f"   Total score: {evaluation_report['evaluation']['total_score']}/{evaluation_report['evaluation']['max_score']}")
        print(f"   Grade: {evaluation_report['evaluation'].get('grade', 'N/A')}")
        
        return evaluation_report
    
    def feedback_phase(self, evaluation_report: Dict[str, Any], extracted_data: Dict[str, Any]) -> str:
        """
        Phase 4: Generate detailed feedback.
        
        Args:
            evaluation_report: Report from evaluation phase
            extracted_data: Original extracted data
            
        Returns:
            Complete feedback text
        """
        print("\n💬 Phase 4: Generating detailed feedback...")
        start_time = time.time()
        
        feedback = self.feedback_generator.generate_complete_feedback(
            evaluation=evaluation_report['evaluation'],
            teacher_data=extracted_data['teacher_key'],
            student_data=extracted_data['student_script']
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Feedback generation completed in {elapsed_time:.2f} seconds")
        
        return feedback
    
    async def run_async(
        self,
        teacher_file_path: str,
        student_file_path: str,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete correction pipeline asynchronously.
        
        Args:
            teacher_file_path: Path to teacher's answer key
            student_file_path: Path to student's script
            save_results: Whether to save results to files
            
        Returns:
            Complete pipeline results
        """
        print("🚀 Starting Automated Paper Correction Pipeline")
        print("="*60)
        
        # Safety Check: Verify API prerequisites
        print("\n🔍 Checking API prerequisites...")
        all_ok, issues = check_api_prerequisites()
        for issue in issues:
            print(f"   {issue}")
        
        if not all_ok:
            error_msg = "API prerequisites check failed. Please configure GEMINI_API_KEY."
            print(f"\n❌ {error_msg}")
            raise ValueError(error_msg)
        
        print("")
        pipeline_start = time.time()
        
        # Phase 1: Extraction
        extracted_data = await self.extract_phase(teacher_file_path, student_file_path)
        
        # Phase 2: Comparison
        comparison_results = await self.comparison_phase(extracted_data)
        
        # Phase 3: Evaluation
        evaluation_report = self.evaluation_phase(comparison_results, extracted_data)
        
        # Phase 4: Feedback
        feedback = self.feedback_phase(evaluation_report, extracted_data)
        
        # Compile final results
        final_results = {
            "extracted_data": extracted_data,
            "comparison_results": comparison_results,
            "evaluation_report": evaluation_report,
            "feedback": feedback,
            "pipeline_metadata": {
                "comparison_method": self.comparison_method,
                "ai_feedback_enabled": self.use_ai_feedback,
                "total_marks": self.total_marks
            }
        }
        
        # Save results if requested
        if save_results:
            self._save_results(final_results, student_file_path)
        
        pipeline_elapsed = time.time() - pipeline_start
        print("\n" + "="*60)
        print(f"✅ Pipeline completed successfully in {pipeline_elapsed:.2f} seconds")
        print("="*60)
        
        return final_results
    
    def run_sync(
        self,
        teacher_file_path: str,
        student_file_path: str,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for running the pipeline.
        
        Args:
            teacher_file_path: Path to teacher's answer key
            student_file_path: Path to student's script
            save_results: Whether to save results to files
            
        Returns:
            Complete pipeline results
        """
        return asyncio.run(self.run_async(teacher_file_path, student_file_path, save_results))
    
    def _save_results(self, results: Dict[str, Any], student_file_path: str) -> None:
        """
        Save pipeline results to files.
        
        Args:
            results: Complete pipeline results
            student_file_path: Path to student file (used for naming)
        """
        print("\n💾 Saving results...")
        
        # Generate filename base from student file
        student_name = Path(student_file_path).stem
        
        # Save full JSON report
        json_path = Path(self.output_dir) / f"{student_name}_evaluation_report.json"
        save_json(results['evaluation_report'], str(json_path))
        print(f"   ✓ Saved evaluation report: {json_path}")
        
        # Save feedback text
        feedback_path = Path(self.output_dir) / f"{student_name}_feedback.txt"
        with open(feedback_path, 'w', encoding='utf-8') as f:
            f.write(results['feedback'])
        print(f"   ✓ Saved feedback: {feedback_path}")
        
        # Save extracted data for reference
        extraction_path = Path(self.output_dir) / f"{student_name}_extracted_data.json"
        save_json(results['extracted_data'], str(extraction_path))
        print(f"   ✓ Saved extracted data: {extraction_path}")
        
        # Save summary
        summary_path = Path(self.output_dir) / f"{student_name}_summary.txt"
        summary = self.evaluator.get_summary(results['evaluation_report']['evaluation'])
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"   ✓ Saved summary: {summary_path}")


def run_correction_pipeline(
    teacher_file_path: str,
    student_file_path: str,
    comparison_method: str = "gemini",
    use_ai_feedback: bool = False,
    total_marks: float = 100.0,
    output_dir: str = "results",
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run the correction pipeline.
    
    Args:
        teacher_file_path: Path to teacher's answer key
        student_file_path: Path to student's script
        comparison_method: Comparison method ("gemini" or "sentence_transformers")
        use_ai_feedback: Whether to use AI for feedback
        total_marks: Total marks for assessment
        output_dir: Directory for results
        save_results: Whether to save results
        
    Returns:
        Complete pipeline results
    """
    pipeline = CorrectionPipeline(
        comparison_method=comparison_method,
        use_ai_feedback=use_ai_feedback,
        total_marks=total_marks,
        output_dir=output_dir
    )
    
    return pipeline.run_sync(teacher_file_path, student_file_path, save_results)
