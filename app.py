"""
Streamlit web application for Automated Paper Correction System.
Provides an intuitive interface for uploading and evaluating student scripts.
"""
import streamlit as st
import os
from pathlib import Path
import time

from pipeline import run_correction_pipeline
from utils import save_uploaded_file, validate_file_extension, ensure_directory_exists, verify_gemini_api_key
import asyncio
import nest_asyncio

# Fix asyncio event loop issues in Streamlit
try:
    nest_asyncio.apply()
except:
    pass


# Page configuration
st.set_page_config(
    page_title="Automated Paper Correction System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display application header."""
    st.title("📝 Automated Paper Correction System")
    st.markdown("""
    Upload teacher's answer key and student's script to automatically evaluate and generate detailed feedback.
    Powered by **Gemini 2.5 Flash** for intelligent document processing and semantic analysis.
    """)
    st.divider()


def display_sidebar():
    """Display sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    # Comparison method selection
    comparison_method = st.sidebar.selectbox(
        "Comparison Method",
        ["gemini", "sentence_transformers"],
        help="Gemini 2.5 Flash (recommended) provides superior accuracy. Sentence Transformers is faster but less accurate."
    )
    
    # AI feedback toggle
    use_ai_feedback = st.sidebar.checkbox(
        "Use AI for Feedback",
        value=False,
        help="Enable Gemini AI for more detailed, personalized feedback. Requires API key."
    )
    
    # Total marks
    total_marks = st.sidebar.number_input(
        "Total Marks",
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Total marks for the assessment"
    )
    
    # Pass threshold
    pass_threshold = st.sidebar.number_input(
        "Pass Threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=5.0,
        help="Minimum percentage required to pass"
    )
    
    st.sidebar.divider()
    
    # API Key configuration
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        gemini_api_key = st.text_input(
            "Gemini API Key (Required)",
            type="password",
            help="Required for Gemini-powered extraction and comparison"
        )
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
            # Verify the key
            is_valid, message = verify_gemini_api_key()
            if is_valid:
                st.success("✅ API key verified")
            else:
                st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Please enter your Gemini API key")
    
    st.sidebar.divider()
    
    # About section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Features:**
        - Multi-page PDF support
        - Gemini 2.5 Flash for extraction
        - AI-powered semantic analysis
        - Detailed feedback generation
        - Structured JSON reports
        
        **Version:** 2.0.0 (Gemini-Powered)
        """)
    
    return comparison_method, use_ai_feedback, total_marks, pass_threshold


def upload_files():
    """Handle file uploads."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Teacher's Answer Key")
        teacher_file = st.file_uploader(
            "Upload teacher's answer key (PDF or Image)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key='teacher_upload'
        )
    
    with col2:
        st.subheader("📝 Student's Script")
        student_file = st.file_uploader(
            "Upload student's script (PDF or Image)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key='student_upload'
        )
    
    return teacher_file, student_file


def display_results(results):
    """Display evaluation results."""
    if not results:
        return
    
    evaluation = results['evaluation_report']['evaluation']
    
    # Summary metrics
    st.header("📊 Evaluation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Score",
            f"{evaluation['total_score']}/{evaluation['max_score']}",
            delta=None
        )
    
    with col2:
        percentage = evaluation['percentage']
        st.metric(
            "Percentage",
            f"{percentage:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Grade",
            evaluation.get('grade', 'N/A'),
            delta=None
        )
    
    with col4:
        status_emoji = "✅" if evaluation['status'] == 'pass' else "❌"
        st.metric(
            "Status",
            f"{status_emoji} {evaluation['status'].upper()}",
            delta=None
        )
    
    st.divider()
    
    # Detailed page-wise results
    st.header("📄 Page-wise Analysis")
    
    for page_score in evaluation['page_scores']:
        with st.expander(f"Page {page_score['page_no']} - Score: {page_score['marks_awarded']}/{page_score['max_marks']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Similarity", f"{page_score['similarity_score']:.2f}%")
                st.progress(page_score['similarity_score'] / 100)
            
            with col2:
                if page_score.get('analysis'):
                    st.write("**Analysis:**")
                    st.info(page_score['analysis'])
    
    st.divider()
    
    # Detailed feedback
    st.header("💬 Detailed Feedback")
    st.text_area(
        "Generated Feedback",
        value=results['feedback'],
        height=400,
        disabled=True
    )
    
    st.divider()
    
    # Download options
    st.header("⬇️ Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download evaluation report as JSON
        import json
        json_str = json.dumps(results['evaluation_report'], indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name="evaluation_report.json",
            mime="application/json"
        )
    
    with col2:
        # Download feedback as text
        st.download_button(
            label="📥 Download Feedback",
            data=results['feedback'],
            file_name="feedback.txt",
            mime="text/plain"
        )
    
    with col3:
        # Download summary
        from evaluation import Evaluator
        evaluator = Evaluator()
        summary = evaluator.get_summary(evaluation)
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )


def process_papers(teacher_file, student_file, comparison_method, use_ai_feedback, total_marks):
    """Process the uploaded papers through the pipeline."""
    try:
        # Verify API key first
        is_valid, message = verify_gemini_api_key()
        if not is_valid:
            st.error(f"❌ {message}")
            st.info("💡 Please add your Gemini API key in the sidebar configuration.")
            return None
        
        # Save uploaded files
        temp_dir = "temp_uploads"
        ensure_directory_exists(temp_dir)
        
        with st.spinner("Saving uploaded files..."):
            teacher_path = save_uploaded_file(teacher_file, temp_dir)
            student_path = save_uploaded_file(student_file, temp_dir)
        
        # Run the correction pipeline
        with st.spinner("Processing documents with Gemini 2.5 Flash... This may take a few moments."):
            progress_bar = st.progress(0)
            
            # Update progress (simulated for user feedback)
            progress_bar.progress(10)
            st.write("✅ Converting PDFs to images...")
            
            progress_bar.progress(25)
            st.write("✅ Extracting text with Gemini...")
            
            # Run pipeline in a way that works with Streamlit
            try:
                results = run_correction_pipeline(
                    teacher_file_path=teacher_path,
                    student_file_path=student_path,
                    comparison_method=comparison_method,
                    use_ai_feedback=use_ai_feedback,
                    total_marks=total_marks,
                    output_dir="results",
                    save_results=False  # Don't save in Streamlit mode
                )
            except RuntimeError as e:
                if "event loop" in str(e).lower():
                    # Handle nested event loop issue
                    st.warning("⚠️ Detected async loop issue, retrying...")
                    import nest_asyncio
                    nest_asyncio.apply()
                    results = run_correction_pipeline(
                        teacher_file_path=teacher_path,
                        student_file_path=student_path,
                        comparison_method=comparison_method,
                        use_ai_feedback=use_ai_feedback,
                        total_marks=total_marks,
                        output_dir="results",
                        save_results=False
                    )
                else:
                    raise
            
            progress_bar.progress(100)
            st.success("✅ Processing completed successfully!")
        
        return results
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
        return None


def main():
    """Main application function."""
    initialize_session_state()
    display_header()
    
    # Sidebar configuration
    comparison_method, use_ai_feedback, total_marks, pass_threshold = display_sidebar()
    
    # File upload section
    teacher_file, student_file = upload_files()
    
    # Process button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process and Evaluate",
            type="primary",
            use_container_width=True,
            disabled=not (teacher_file and student_file)
        )
    
    if not (teacher_file and student_file):
        st.info("👆 Please upload both teacher's answer key and student's script to proceed.")
    
    # Process papers when button is clicked
    if process_button and teacher_file and student_file:
        st.divider()
        results = process_papers(
            teacher_file,
            student_file,
            comparison_method,
            use_ai_feedback,
            total_marks
        )
        
        if results:
            st.session_state.results = results
    
    # Display results if available
    if st.session_state.results:
        st.divider()
        display_results(st.session_state.results)


if __name__ == "__main__":
    main()
