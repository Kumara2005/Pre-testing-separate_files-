"""
Streamlit web application for Automated Paper Correction System.
Updated to support Few-Shot learning with an optional reference paper.
"""
import streamlit as st
import os
from pathlib import Path
import time
import asyncio
import nest_asyncio

from pipeline import run_correction_pipeline
from utils import save_uploaded_file, validate_file_extension, ensure_directory_exists, verify_gemini_api_key

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
    Upload teacher's answer key and student's script to automatically evaluate. 
    **New:** Upload an optional manually corrected reference paper to align AI grading style (Few-Shot Learning).
    """)
    st.divider()

def display_sidebar():
    """Display sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    comparison_method = st.sidebar.selectbox(
        "Comparison Method",
        ["gemini", "sentence_transformers"],
        help="Gemini 2.5 Flash (recommended) provides superior accuracy."
    )
    
    use_ai_feedback = st.sidebar.checkbox(
        "Use AI for Feedback",
        value=True,
        help="Enable Gemini AI for personalized feedback."
    )
    
    total_marks = st.sidebar.number_input(
        "Total Marks",
        min_value=10.0,
        max_value=1000.0,
        value=100.0,
        step=10.0
    )
    
    st.sidebar.divider()
    
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        gemini_api_key = st.text_input(
            "Gemini API Key (Required)",
            type="password"
        )
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
            is_valid, message = verify_gemini_api_key()
            if is_valid:
                st.success("✅ API key verified")
            else:
                st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Please enter your API key")
    
    return comparison_method, use_ai_feedback, total_marks

def upload_files():
    """Handle file uploads (including optional reference)."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📄 Teacher's Key")
        teacher_file = st.file_uploader(
            "Official answer key",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key='teacher_upload'
        )
    
    with col2:
        st.subheader("📝 Student's Script")
        student_file = st.file_uploader(
            "Student's written paper",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key='student_upload'
        )

    with col3:
        st.subheader("💡 Reference (Optional)")
        reference_file = st.file_uploader(
            "Manually corrected sample",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key='reference_upload',
            help="Upload a paper already corrected by a teacher to guide the AI's grading style."
        )
    
    return teacher_file, student_file, reference_file

def display_results(results):
    """Display evaluation results."""
    if not results:
        return
    
    evaluation = results['evaluation_report']['evaluation']
    
    st.header("📊 Evaluation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Score", f"{evaluation['total_score']}/{evaluation['max_score']}")
    col2.metric("Percentage", f"{evaluation['percentage']:.2f}%")
    col3.metric("Grade", evaluation.get('grade', 'N/A'))
    col4.metric("Status", f"{evaluation['status'].upper()}")
    
    st.divider()
    
    st.header("📄 Page-wise Analysis")
    for page_score in evaluation['page_scores']:
        with st.expander(f"Page {page_score['page_no']} - Score: {page_score['marks_awarded']}/{page_score['max_marks']}"):
            col_a, col_b = st.columns([1, 2])
            col_a.metric("Similarity", f"{page_score['similarity_score']:.2f}%")
            col_a.progress(page_score['similarity_score'] / 100)
            if page_score.get('analysis'):
                col_b.write("**Analysis:**")
                col_b.info(page_score['analysis'])
    
    st.divider()
    st.header("💬 Detailed Feedback")
    st.text_area("Generated Feedback", value=results['feedback'], height=300, disabled=True)

def process_papers(teacher_file, student_file, reference_file, comparison_method, use_ai_feedback, total_marks):
    """Process files through the pipeline."""
    try:
        is_valid, message = verify_gemini_api_key()
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        temp_dir = "temp_uploads"
        ensure_directory_exists(temp_dir)
        
        with st.spinner("Saving uploaded files..."):
            teacher_path = save_uploaded_file(teacher_file, temp_dir)
            student_path = save_uploaded_file(student_file, temp_dir)
            # Handle optional reference file
            reference_path = save_uploaded_file(reference_file, temp_dir) if reference_file else None
        
        with st.spinner("Processing with Gemini 2.5 Flash... This involves OCR and Semantic Analysis."):
            results = run_correction_pipeline(
                teacher_file_path=teacher_path,
                student_file_path=student_path,
                reference_file_path=reference_path, # Pass the reference path
                comparison_method=comparison_method,
                use_ai_feedback=use_ai_feedback,
                total_marks=total_marks,
                output_dir="results",
                save_results=False
            )
        
        st.success("✅ Evaluation Complete!")
        return results
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        return None

def main():
    initialize_session_state()
    display_header()
    
    comparison_method, use_ai_feedback, total_marks = display_sidebar()
    teacher_file, student_file, reference_file = upload_files()
    
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process and Evaluate",
            type="primary",
            use_container_width=True,
            disabled=not (teacher_file and student_file)
        )
    
    if process_button:
        results = process_papers(
            teacher_file, student_file, reference_file,
            comparison_method, use_ai_feedback, total_marks
        )
        if results:
            st.session_state.results = results
    
    if st.session_state.results:
        display_results(st.session_state.results)

if __name__ == "__main__":
    main()