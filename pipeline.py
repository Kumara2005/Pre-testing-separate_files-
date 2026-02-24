"""
Pipeline orchestrator that connects all modules into a seamless workflow.
Coordinates extraction, comparison, evaluation, and feedback generation.
Updated to support few-shot learning with an optional reference paper.
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
        student_file_path: str,
        reference_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Extract text from documents using sequential safe-mode.
        Now includes optional reference paper extraction.
        """
        print("📄 Phase 1: Sequential Extraction (Free Tier Safe Mode)...")
        start_time = time.time()
        
        extracted_data = await extract_documents(
            teacher_file_path, 
            student_file_path, 
            reference_file_path
        )
        
        # Validate that we actually got text
        t_text = sum(len(p.get('content', '')) for p in extracted_data['teacher_key']['pages'])
        s_text = sum(len(p.get('content', '')) for p in extracted_data['student_script']['pages'])
        
        if t_text == 0 or s_text == 0:
            print("⚠️ WARNING: Extraction returned empty text. Check API Key/Connection.")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Extraction completed in {elapsed_time:.2f} seconds")
        return extracted_data
    
    async def comparison_phase(self, extracted_data: Dict[str, Any]) -> list:
        """
        Phase 2: Compare student script with teacher's key using Few-Shot alignment.
        """
        print("\n🔍 Phase 2: Comparing student answers with few-shot calibration...")
        start_time = time.time()
        
        teacher_data = extracted_data['teacher_key']
        student_data = extracted_data['student_script']
        reference_data = extracted_data.get('reference_paper') # Might be None
        
        comparison_results = await self.comparator.compare_documents(
            teacher_data, 
            student_data, 
            reference_data
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Comparison completed in {elapsed_time:.2f} seconds")
        return comparison_results
    
    def evaluation_phase(self, comparison_results: list, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Evaluate and generate scores.
        """
        print("\n📊 Phase 3: Evaluating and generating scores...")
        teacher_file = extracted_data['teacher_key'].get('file_name', 'teacher_key.pdf')
        student_file = extracted_data['student_script'].get('file_name', 'student_script.pdf')
        
        evaluation_report = self.evaluator.generate_evaluation_report(
            comparison_results=comparison_results,
            teacher_file=teacher_file,
            student_file=student_file
        )
        return evaluation_report
    
    def feedback_phase(self, evaluation_report: Dict[str, Any], extracted_data: Dict[str, Any]) -> str:
        """
        Phase 4: Generate detailed feedback.
        """
        print("\n💬 Phase 4: Generating detailed feedback...")
        feedback = self.feedback_generator.generate_complete_feedback(
            evaluation=evaluation_report['evaluation'],
            teacher_data=extracted_data['teacher_key'],
            student_data=extracted_data['student_script']
        )
        return feedback
    
    async def run_async(
        self,
        teacher_file_path: str,
        student_file_path: str,
        reference_file_path: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete correction pipeline asynchronously with few-shot support.
        """
        print("🚀 Starting Automated Paper Correction Pipeline (Few-Shot Mode)")
        print("="*60)
        
        # Safety Check: Verify API prerequisites
        all_ok, issues = check_api_prerequisites()
        if not all_ok:
            raise ValueError("API prerequisites check failed.")
        
        pipeline_start = time.time()
        
        # Phase 1: Extraction (Includes Reference if provided)
        extracted_data = await self.extract_phase(teacher_file_path, student_file_path, reference_file_path)
        
        # Phase 2: Comparison (Few-Shot Calibration happens here)
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
                "total_marks": self.total_marks,
                "few_shot_enabled": reference_file_path is not None
            }
        }
        
        if save_results:
            self._save_results(final_results, student_file_path)
        
        pipeline_elapsed = time.time() - pipeline_start
        print("\n" + "="*60)
        print(f"✅ Pipeline completed in {pipeline_elapsed:.2f} seconds")
        return final_results
    
    def run_sync(
        self,
        teacher_file_path: str,
        student_file_path: str,
        reference_file_path: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper."""
        return asyncio.run(self.run_async(teacher_file_path, student_file_path, reference_file_path, save_results))
    
    def _save_results(self, results: Dict[str, Any], student_file_path: str) -> None:
        """Save results to files."""
        student_name = Path(student_file_path).stem
        save_json(results['evaluation_report'], str(Path(self.output_dir) / f"{student_name}_report.json"))
        
        with open(Path(self.output_dir) / f"{student_name}_feedback.txt", 'w', encoding='utf-8') as f:
            f.write(results['feedback'])


def run_correction_pipeline(
    teacher_file_path: str,
    student_file_path: str,
    reference_file_path: Optional[str] = None,
    comparison_method: str = "gemini",
    use_ai_feedback: bool = False,
    total_marks: float = 100.0,
    output_dir: str = "results",
    save_results: bool = True
) -> Dict[str, Any]:
    """Convenience function to run the correction pipeline with Reference support."""
    pipeline = CorrectionPipeline(
        comparison_method=comparison_method,
        use_ai_feedback=use_ai_feedback,
        total_marks=total_marks,
        output_dir=output_dir
    )
    
    return pipeline.run_sync(teacher_file_path, student_file_path, reference_file_path, save_results)