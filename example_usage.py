"""
Example usage script for Automated Paper Correction System.
Demonstrates how to use the Gemini-powered pipeline programmatically.
"""

from pipeline import run_correction_pipeline, CorrectionPipeline
import asyncio


def example_basic_usage():
    """Basic usage example with Gemini (default settings)."""
    print("=== Basic Usage Example (Gemini-Powered) ===\n")
    
    # Run with default Gemini settings
    results = run_correction_pipeline(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        save_results=True
    )
    
    # Access results
    evaluation = results['evaluation_report']['evaluation']
    print(f"Score: {evaluation['total_score']}/{evaluation['max_score']}")
    print(f"Grade: {evaluation['grade']}")
    print(f"Status: {evaluation['status']}")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # Run with custom settings
    results = run_correction_pipeline(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        comparison_method="gemini",  # Default, but can use "sentence_transformers"
        use_ai_feedback=True,  # Enable AI-generated detailed feedback
        total_marks=150.0,  # Custom total marks
        output_dir="custom_results",  # Custom output directory
        save_results=True
    )
    
    print(f"Results saved to: custom_results/")


async def example_async_usage():
    """Example using async pipeline for better performance."""
    print("\n=== Async Usage Example (Gemini-Powered) ===\n")
    
    # Create pipeline instance with Gemini
    pipeline = CorrectionPipeline(
        comparison_method="gemini",
        use_ai_feedback=True,
        total_marks=100.0,
        output_dir="results"
    )
    
    # Run asynchronously
    results = await pipeline.run_async(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        save_results=True
    )
    
    # Access specific components
    print("\nExtracted Pages:")
    print(f"Teacher: {results['extracted_data']['teacher_key']['total_pages']} pages")
    print(f"Student: {results['extracted_data']['student_script']['total_pages']} pages")
    
    print("\nComparison Results:")
    for i, comp in enumerate(results['comparison_results'], 1):
        print(f"Page {i}: {comp['similarity']*100:.2f}% similarity")
    
    print("\nEvaluation:")
    evaluation = results['evaluation_report']['evaluation']
    print(f"Final Score: {evaluation['total_score']}/{evaluation['max_score']}")
    print(f"Grade: {evaluation['grade']}")
    
    print("\nFeedback Preview:")
    print(results['feedback'][:200] + "...")


def example_batch_processing():
    """Example of processing multiple students."""
    print("\n=== Batch Processing Example ===\n")
    
    # List of student files to process
    students = [
        ("student1_script.pdf", "Student 1"),
        ("student2_script.pdf", "Student 2"),
        ("student3_script.pdf", "Student 3"),
    ]
    
    teacher_key = "path/to/teacher_answer_key.pdf"
    
    results_summary = []
    
    for student_file, student_name in students:
        print(f"\nProcessing {student_name}...")
        
        try:
            results = run_correction_pipeline(
                teacher_file_path=teacher_key,
                student_file_path=student_file,
                save_results=True
            )
            
            evaluation = results['evaluation_report']['evaluation']
            results_summary.append({
                "name": student_name,
                "score": evaluation['total_score'],
                "grade": evaluation['grade'],
                "status": evaluation['status']
            })
            
            print(f"✓ {student_name}: {evaluation['total_score']} ({evaluation['grade']})")
        
        except Exception as e:
            print(f"✗ Error processing {student_name}: {e}")
            results_summary.append({
                "name": student_name,
                "error": str(e)
            })
    
    # Display summary
    print("\n=== Batch Processing Summary ===")
    for result in results_summary:
        if 'error' in result:
            print(f"{result['name']}: ERROR - {result['error']}")
        else:
            print(f"{result['name']}: {result['score']} - {result['grade']} - {result['status']}")


def example_accessing_detailed_results():
    """Example of accessing detailed results from the pipeline."""
    print("\n=== Accessing Detailed Results Example ===\n")
    
    results = run_correction_pipeline(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        save_results=False  # Don't save to disk
    )
    
    # Access extracted text
    print("1. Extracted Text:")
    teacher_pages = results['extracted_data']['teacher_key']['pages']
    print(f"   First page content preview: {teacher_pages[0]['content'][:100]}...")
    
    # Access comparison results
    print("\n2. Comparison Results:")
    for comp in results['comparison_results']:
        print(f"   Page {comp['student_page_no']}: "
              f"Similarity = {comp['similarity']*100:.2f}%")
    
    # Access evaluation details
    print("\n3. Evaluation Details:")
    evaluation = results['evaluation_report']['evaluation']
    print(f"   Total Score: {evaluation['total_score']}/{evaluation['max_score']}")
    print(f"   Percentage: {evaluation['percentage']}%")
    print(f"   Grade: {evaluation['grade']}")
    
    # Access page-wise scores
    print("\n4. Page-wise Scores:")
    for page in evaluation['page_scores']:
        print(f"   Page {page['page_no']}: {page['marks_awarded']}/{page['max_marks']} "
              f"(Similarity: {page['similarity_score']}%)")
    
    # Access feedback
    print("\n5. Feedback:")
    print(results['feedback'])


def example_with_gemini():
    """Example using Gemini API for all features (Recommended)."""
    print("\n=== Full Gemini API Example (Recommended) ===\n")
    print("Uses Gemini 2.5 Flash for both extraction and comparison\n")
    
    results = run_correction_pipeline(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        comparison_method="gemini",  # Use Gemini for comparison
        use_ai_feedback=True,  # Use AI for feedback generation
        save_results=True
    )
    
    # Gemini provides more detailed analysis
    evaluation = results['evaluation_report']['evaluation']
    print(f"Score: {evaluation['total_score']}/{evaluation['max_score']}")
    print(f"\nDetailed AI-generated feedback available in results")


def example_with_fallback():
    """Example using Sentence Transformers as fallback."""
    print("\n=== Fallback Method Example ===\n")
    print("Uses Sentence Transformers (no API key required)\n")
    
    results = run_correction_pipeline(
        teacher_file_path="path/to/teacher_answer_key.pdf",
        student_file_path="path/to/student_script.pdf",
        comparison_method="sentence_transformers",  # Fallback method
        use_ai_feedback=False,  # No AI feedback
        save_results=True
    )
    
    evaluation = results['evaluation_report']['evaluation']
    print(f"Score: {evaluation['total_score']}/{evaluation['max_score']}")


if __name__ == "__main__":
    print("Automated Paper Correction System - Usage Examples")
    print("v2.0 - Powered by Gemini 2.5 Flash")
    print("="*60)
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_custom_configuration()
    # asyncio.run(example_async_usage())
    # example_batch_processing()
    # example_accessing_detailed_results()
    # example_with_gemini()  # Recommended
    # example_with_fallback()
    
    print("\n" + "="*60)
    print("To run these examples, uncomment the desired function call")
    print("and update the file paths to your actual PDF files.")
    print("Make sure GEMINI_API_KEY is set in your .env file.")
    print("="*60)
