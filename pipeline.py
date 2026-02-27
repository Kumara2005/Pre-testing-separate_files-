"""
Pipeline orchestrator that connects all modules into a seamless workflow.
Coordinates extraction, comparison, evaluation, and feedback generation.
Updated with Leniency Logic (0.55 threshold) and Global Context Mapping.
"""
import asyncio
import time
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


# Module imports
from extraction import DocumentExtractor, extract_documents
from compare import SemanticComparator
from evaluation import Evaluator
from feedback import FeedbackGenerator
from utils import save_json, ensure_directory_exists, check_api_prerequisites


class CorrectionPipeline:
    """Orchestrates the complete paper correction workflow."""
    
    def __init__(
        self,
        comparison_method: str = "gemini",
        use_ai_feedback: bool = False,
        total_marks: float = 100.0,
        output_dir: str = "results"
    ):
        """Initialize core components of the correction pipeline."""
        self.comparison_method = comparison_method
        self.use_ai_feedback = use_ai_feedback
        self.total_marks = total_marks
        self.output_dir = output_dir
        
        self.extractor = DocumentExtractor()
        self.comparator = SemanticComparator(method=comparison_method)
        self.evaluator = Evaluator(total_marks=total_marks)
        self.feedback_generator = FeedbackGenerator(use_ai=use_ai_feedback)
        
        ensure_directory_exists(output_dir)
    
    async def extract_phase(self, teacher_file_path: str, student_file_path: str, reference_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Phase 1: Extract text and parse metadata (Name and Roll No)."""
        extracted_data = await extract_documents(teacher_file_path, student_file_path, reference_file_path)
        
        if 'student_script' not in extracted_data:
            raise KeyError("Critical Error: 'student_script' missing from extraction results.")

        student_script = extracted_data['student_script']
        extracted_data['student_name'] = "Unknown"
        extracted_data['roll_no'] = "Unknown"

        # Parsing student metadata from headers to identify the student
        for page in student_script['pages']:
            raw = page.get('raw_text', '')
            if "METADATA_NAME:" in raw and "CONTENT:" in raw:
                try:
                    name = raw.split("METADATA_NAME:")[1].split("METADATA_ROLL:")[0].strip()
                    roll = raw.split("METADATA_ROLL:")[1].split("CONTENT:")[0].strip()
                    # Clean the page content by removing header tags
                    page['content'] = raw.split("CONTENT:")[1].strip()
                    
                    extracted_data['student_name'] = name if name.lower() != "unknown" else "Unknown"
                    extracted_data['roll_no'] = roll if roll.lower() != "unknown" else "Unknown"
                except (IndexError, ValueError):
                    pass
        return extracted_data

    async def run_async(self, t_path: str, s_path: str, r_path: Optional[str] = None, save_results: bool = True, subject: str = "General") -> Dict[str, Any]:
        """Main execution flow with Global Master Key Mapping and Leniency Logic."""
        pipeline_start = time.time()
        
        # Phase 1: Extraction
        extracted_data = await self.extract_phase(t_path, s_path, r_path)
        
        # --- DUAL-MODE LOGIC & THRESHOLD CALIBRATION ---
       # Inside pipeline.py -> run_async method

    # --- DUAL-MODE LOGIC ---
        if subject == "Language":
            eval_mode = "language"
            threshold = 0.85 # Strict for Grammar/Creativity
        else:
            eval_mode = "technical"
            threshold = 0.50 # Lenient for Logic/Keywords

    # ... remaining pipeline logic ...

        # --- GLOBAL CONTEXT MAPPING (The Fix) ---
        # Consolidate all teacher pages into one Master Key. 
        # Resolves cases where Teacher has 8 Qs on Page 1 but Student has 2 Qs per page.
        full_teacher_key = "\n\n".join([p.get("content", "") for p in extracted_data['teacher_key']['pages']])
        
        print(f"🔍 Phase 2: Comparing answers using '{eval_mode}' mode with Global Key Mapping.")

        # Phase 2: Comparison (Passing the Master Key and leniency threshold)
        comparison_results = await self.comparator.compare_documents(
            teacher_data=extracted_data['teacher_key'], 
            student_data=extracted_data['student_script'], 
            subject=subject,
            mode=eval_mode,
            threshold=threshold,
            master_key_content=full_teacher_key 
        )
        
        # Phase 3: Evaluation (Dynamic question-wise summation)
        evaluation_report = self.evaluator.generate_evaluation_report(
            comparison_results=comparison_results,
            teacher_file=extracted_data['teacher_key'].get('file_name'),
            student_file=extracted_data['student_script'].get('file_name'),
            student_info={
                "name": extracted_data.get('student_name'),
                "roll_no": extracted_data.get('roll_no')
            }
        )
        
        # Phase 4: Feedback
        feedback = self.feedback_generator.generate_complete_feedback(
            evaluation=evaluation_report['evaluation'],
            teacher_data=extracted_data['teacher_key'],
            student_data=extracted_data['student_script']
        )
        
        final_results = {
            "extracted_data": extracted_data,
            "evaluation_report": evaluation_report,
            "feedback": feedback,
            "pipeline_metadata": {
                "subject": subject,
                "eval_mode": eval_mode,
                "elapsed_time": time.time() - pipeline_start
            }
        }
        
        if save_results:
            self._save_results(final_results, s_path)
        
        return final_results

    def run_sync(self, t_path: str, s_path: str, r_path: Optional[str] = None, save_results: bool = True, subject: str = "General"):
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(t_path, s_path, r_path, save_results, subject))

   # Inside pipeline.py

    def _save_results(self, results: Dict[str, Any], s_path: str):
        """Saves report using a sanitized student name and roll number."""
        # Extract raw metadata
        raw_name = results['extracted_data'].get('student_name', 'Unknown')
        raw_roll = results['extracted_data'].get('roll_no', 'Unknown')
        
        # FIX: Sanitize the filename to remove newlines, colons, and extra spaces
        def sanitize(text):
            # Remove newlines, colons, and any other illegal characters
            text = text.replace('\n', '').replace('\r', '').replace(':', '_')
            # Only allow alphanumeric, underscores, and hyphens
            return re.sub(r'[^\w\-_\. ]', '', text).strip()

        name = sanitize(raw_name)
        roll = sanitize(raw_roll)
        
        # Construct a clean filename
        filename = f"{name}_{roll}_report.json".replace(" ", "_")
        
        save_json(results['evaluation_report'], str(Path(self.output_dir) / filename))


def run_correction_pipeline(teacher_file_path: str, student_file_path: str, reference_file_path: Optional[str] = None, total_marks: float = 100.0, subject: str = "General"):
    """Global entry point."""
    pipeline = CorrectionPipeline(total_marks=total_marks)
    return pipeline.run_sync(teacher_file_path, student_file_path, reference_file_path, subject=subject)