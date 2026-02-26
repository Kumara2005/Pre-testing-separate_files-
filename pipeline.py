"""
Pipeline orchestrator that connects all modules into a seamless workflow.
Coordinates extraction, comparison, evaluation, and feedback generation.
Updated to support student name extraction and subject-aware evaluation.
"""
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

# Internal module imports (Ensure these files exist in your directory)
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
        """Initialize the correction pipeline."""
        self.comparison_method = comparison_method
        self.use_ai_feedback = use_ai_feedback
        self.total_marks = total_marks
        self.output_dir = output_dir
        
        # Initialize core components
        self.extractor = DocumentExtractor()
        self.comparator = SemanticComparator(method=comparison_method)
        self.evaluator = Evaluator(total_marks=total_marks)
        self.feedback_generator = FeedbackGenerator(use_ai=use_ai_feedback)
        
        ensure_directory_exists(output_dir)
    
    async def extract_phase(
        self,
        teacher_file_path: str,
        student_file_path: str,
        reference_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Phase 1: Extract text and parse metadata like Student Name."""
        print("📄 Phase 1: Sequential Extraction (Free Tier Safe Mode)...")
        start_time = time.time()
        
        extracted_data = await extract_documents(
            teacher_file_path, 
            student_file_path, 
            reference_file_path
        )
        
        # Post-extraction logic to parse student name from the raw AI response
        student_data = extracted_data['student_script']
        for page in student_data['pages']:
            raw = page.get('raw_response', '')
            if "STUDENT_NAME:" in raw and "CONTENT:" in raw:
                # Extract name from the page 1 header logic
                name_part = raw.split("STUDENT_NAME:")[1].split("CONTENT:")[0].strip()
                page['content'] = raw.split("CONTENT:")[1].strip()
                
                # Assign extracted name to metadata if found
                if name_part and name_part.lower() != "unknown":
                    extracted_data['student_name_from_sheet'] = name_part

        # Check for empty results
        t_text = sum(len(p.get('content', '')) for p in extracted_data['teacher_key']['pages'])
        s_text = sum(len(p.get('content', '')) for p in extracted_data['student_script']['pages'])
        
        if t_text == 0 or s_text == 0:
            print("⚠️ WARNING: Extraction returned empty text. Check API Key.")
        
        print(f"✅ Extraction completed in {time.time() - start_time:.2f} seconds")
        return extracted_data
    
    async def comparison_phase(self, extracted_data: Dict[str, Any], subject: str = "General") -> List[Dict[str, Any]]:
        """Phase 2: Perform subject-aware semantic comparison."""
        print(f"\n🔍 Phase 2: Comparing answers for {subject}...")
        
        comparison_results = await self.comparator.compare_documents(
            extracted_data['teacher_key'], 
            extracted_data['student_script'], 
            subject,
            extracted_data.get('reference_paper')
        )
        return comparison_results
    
    def evaluation_phase(self, comparison_results: List[Dict[str, Any]], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Finalize scoring and generate reports."""
        print("\n📊 Phase 3: Evaluating and generating scores...")
        
        return self.evaluator.generate_evaluation_report(
            comparison_results=comparison_results,
            teacher_file=extracted_data['teacher_key'].get('file_name', 'teacher.pdf'),
            student_file=extracted_data['student_script'].get('file_name', 'student.pdf')
        )
    
    def feedback_phase(self, evaluation_report: Dict[str, Any], extracted_data: Dict[str, Any]) -> str:
        """Phase 4: Generate detailed feedback."""
        print("\n💬 Phase 4: Generating detailed feedback...")
        
        return self.feedback_generator.generate_complete_feedback(
            evaluation=evaluation_report['evaluation'],
            teacher_data=extracted_data['teacher_key'],
            student_data=extracted_data['student_script']
        )
    
    async def run_async(
        self,
        teacher_file_path: str,
        student_file_path: str,
        reference_file_path: Optional[str] = None,
        save_results: bool = True,
        subject: str = "General"
    ) -> Dict[str, Any]:
        """Main asynchronous execution flow."""
        print(f"🚀 Initializing Pipeline | Subject: {subject}")
        print("="*60)
        
        all_ok, _ = check_api_prerequisites()
        if not all_ok:
            raise ValueError("API prerequisites check failed.")
        
        pipeline_start = time.time()
        
        # Step 1: Extraction
        extracted_data = await self.extract_phase(teacher_file_path, student_file_path, reference_file_path)
        
        # Step 2: Comparison (Using Subject for Temperature control)
        comparison_results = await self.comparison_phase(extracted_data, subject)
        
        # Step 3: Evaluation
        evaluation_report = self.evaluation_phase(comparison_results, extracted_data)
        
        # Step 4: Feedback
        feedback = self.feedback_phase(evaluation_report, extracted_data)
        
        final_results = {
            "extracted_data": extracted_data,
            "comparison_results": comparison_results,
            "evaluation_report": evaluation_report,
            "feedback": feedback,
            "pipeline_metadata": {
                "subject": subject,
                "total_marks": self.total_marks,
                "few_shot": reference_file_path is not None
            }
        }
        
        if save_results:
            self._save_results(final_results, student_file_path)
        
        print(f"\n✅ Pipeline completed in {time.time() - pipeline_start:.2f} seconds")
        return final_results
    
    def run_sync(self, t_path, s_path, r_path=None, save_results=True, subject="General"):
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(t_path, s_path, r_path, save_results, subject))
    
    def _save_results(self, results: Dict[str, Any], student_file_path: str) -> None:
        """Saves JSON and Text results to the results directory."""
        # Use extracted name if available, else filename
        name_found = results['extracted_data'].get('student_name_from_sheet')
        student_name = name_found if name_found else Path(student_file_path).stem
        
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
    save_results: bool = True,
    subject: str = "General"
) -> Dict[str, Any]:
    """External entry point for the pipeline."""
    pipeline = CorrectionPipeline(
        comparison_method=comparison_method,
        use_ai_feedback=use_ai_feedback,
        total_marks=total_marks,
        output_dir=output_dir
    )
    
    return pipeline.run_sync(teacher_file_path, student_file_path, reference_file_path, save_results, subject)