import streamlit as st
import os
from pipeline import run_correction_pipeline
from database import ResultsDB 
from evaluation import Evaluator 
# REQUIREMENT: Import utilities to handle file paths properly
from utils import save_uploaded_file, ensure_directory_exists 

st.set_page_config(page_title="Academic Evaluation System", page_icon="📝", layout="wide")

def display_results(results):
    """Renders a clean student-facing dashboard with Accuracy metric."""
    if not results: return
    
    evaluation = results['evaluation_report']['evaluation']
    # Initialize the Evaluator to access summary logic
    evaluator = Evaluator(total_marks=evaluation['max_score'])
    
    st.subheader("🏁 Executive Evaluation Summary")
    
    # SAFETY: Calculation for extraction accuracy across all pages
    page_scores = evaluation.get('page_scores', [])
    if page_scores:
        avg_accuracy = sum(p.get('extraction_confidence', 0) for p in page_scores) / len(page_scores)
    else:
        avg_accuracy = 0.0
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: 
        st.metric("AI Total Score", f"{evaluation['total_score']}/{evaluation['max_score']}")
    with kpi2: 
        # REQUIREMENT: Display Accuracy of evaluation process
        st.metric("Evaluation Accuracy", f"{round(avg_accuracy, 2)}%")
    with kpi3: 
        st.metric("Grade", evaluation.get('grade', 'N/A'))
    with kpi4:
        status_val = evaluation.get('status', 'FAIL').upper()
        label = "🟢 PASS" if status_val == "PASS" else "🔴 FAIL"
        st.markdown(f"**Status**\n### {label}")

    st.markdown("---")
    # REQUIREMENT: Clear question-wise summary
    st.markdown(evaluator.get_summary(evaluation))
    st.markdown("---")
    st.info(results.get('feedback', 'No summary feedback available.'))

def main():
    db = ResultsDB()
    if 'results' not in st.session_state: 
        st.session_state.results = None

    st.title("📝 Automated Paper Correction System")

    # SIDEBAR: Optimized per requirements (No API Key or Fallback Marks)
    st.sidebar.header("⚙️ Evaluation Settings")
    
    # REQUIREMENT: Two options - Language & Other subjects
    subject_type = st.sidebar.selectbox("Subject Category", ["Language", "Other Subjects"])
    
    col1, col2 = st.columns(2)
    with col1: 
        t_file = st.file_uploader("Teacher Key", type=['pdf', 'png', 'jpg'])
    with col2: 
        s_file = st.file_uploader("Student Script", type=['pdf', 'png', 'jpg'])

    if st.button("🚀 Run Evaluation Pipeline", type="primary", use_container_width=True):
        if t_file and s_file:
            # FIX: Create temp directory and save files to get valid paths
            temp_dir = "temp_uploads"
            ensure_directory_exists(temp_dir)
            
            t_path = save_uploaded_file(t_file, temp_dir)
            s_path = save_uploaded_file(s_file, temp_dir)
            
            with st.spinner("Analyzing Answer Mapping..."):
                # Pass strings (paths) to the pipeline, not UploadedFile objects
                results = run_correction_pipeline(
                    teacher_file_path=t_path, 
                    student_file_path=s_path, 
                    subject=subject_type 
                )
                if results: 
                    st.session_state.results = results
                    st.rerun() 

    if st.session_state.results:
        display_results(st.session_state.results)

if __name__ == "__main__":
    main()