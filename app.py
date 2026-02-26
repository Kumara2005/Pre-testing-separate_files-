import streamlit as st
import os
import pandas as pd
from pipeline import run_correction_pipeline
from utils import save_uploaded_file, ensure_directory_exists, verify_gemini_api_key
from database import ResultsDB 

# Page configuration for a professional wide-layout dashboard
st.set_page_config(page_title="Academic Evaluation System", page_icon="📝", layout="wide")

def display_results(results):
    """
    Renders a clean report with rubric-based mapping.
    """
    if not results: return
    
    evaluation = results['evaluation_report']['evaluation']
    
    st.subheader("🏁 Executive Evaluation Summary")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.metric("AI Total Score", f"{evaluation['total_score']}/{evaluation['max_score']}")
    with kpi2: st.metric("Accuracy", f"{evaluation['percentage']}%")
    with kpi3: st.metric("Grade", evaluation.get('grade', 'N/A'))
    with kpi4:
        status = evaluation['status'].upper()
        label = "🟢 PASS" if status == "PASS" else "🔴 FAIL"
        st.markdown(f"**Status**\n### {label}")

    st.markdown("### 📝 Actionable Feedback")
    st.info(results.get('feedback', 'No summary feedback available.'))

    st.divider()

    # Rubric-Based Detailed Breakdown
    with st.expander("🔍 View Question-Wise Rubric Mapping (Part A & B)"):
        for page in evaluation['page_scores']:
            st.markdown(f"#### 📑 Page {page['page_no']} Technical Analysis")
            # The AI Analysis now contains the mapping for 2-mark and 16-mark questions
            st.write(page.get('analysis', 'No technical notes provided.'))
            st.write("---")

def main():
    db = ResultsDB()
    if 'results' not in st.session_state: 
        st.session_state.results = None

    st.title("📝 Automated Paper Correction System")
    st.markdown("Evaluate student scripts with rubric-based mapping (Part A/B) and calibration tracking.")

    # SIDEBAR CONFIGURATION
    st.sidebar.header("⚙️ Configuration")
    subject_choice = st.sidebar.selectbox("Subject", ["English", "Tamil", "Maths", "Science", "Social"])
    total_marks = st.sidebar.number_input("Total Marks", value=100.0)
    
    with st.sidebar.expander("🔑 API Key"):
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
            is_valid, _ = verify_gemini_api_key()
            if is_valid: 
                st.success("Key Verified")

    # UPLOADS SECTION
    col1, col2, col3 = st.columns(3)
    with col1: 
        t_file = st.file_uploader("Teacher Key", type=['pdf', 'png', 'jpg'])
    with col2: 
        s_file = st.file_uploader("Student Script", type=['pdf', 'png', 'jpg'])
    with col3: 
        r_file = st.file_uploader("Reference (Optional)", type=['pdf', 'png', 'jpg'])

    # EXECUTION PIPELINE
    if st.button("🚀 Run Evaluation Pipeline", type="primary", use_container_width=True):
        if t_file and s_file:
            temp_dir = "temp_uploads"
            ensure_directory_exists(temp_dir)
            t_path = save_uploaded_file(t_file, temp_dir)
            s_path = save_uploaded_file(s_file, temp_dir)
            r_path = save_uploaded_file(r_file, temp_dir) if r_file else None

            with st.spinner("Analyzing Answer Mapping & Mark Distribution..."):
                results = run_correction_pipeline(
                    teacher_file_path=t_path,
                    student_file_path=s_path,
                    reference_file_path=r_path,
                    total_marks=total_marks,
                    subject=subject_choice 
                )
                
                if results:
                    st.session_state.results = results
                    st.success("AI Evaluation Complete. Proceed to Calibration below.")

    # DISPLAY AREA & CALIBRATION
    if st.session_state.results:
        display_results(st.session_state.results)
        
        st.divider()
        st.subheader("⚖️ Teacher Calibration & Saving")
        c1, c2 = st.columns(2)
        
        eval_data = st.session_state.results['evaluation_report']['evaluation']
        ai_score = eval_data['total_score']
        
        with c1:
            st.write(f"**AI Predicted Score:** {ai_score}")
            teacher_score = st.number_input("Enter Actual Teacher Score (from manual correction)", value=float(ai_score))
        
        with c2:
            variance = teacher_score - ai_score
            st.metric("Score Variance", f"{variance:+.2f}", help="Difference between Teacher and AI marks.")
            
        if st.button("💾 Finalize and Save to History"):
            detected_name = st.session_state.results['extracted_data'].get('student_name_from_sheet', "Unknown Student")
            
            # Save using the updated DB method that tracks variance
            db.insert_result(
                name=detected_name, 
                subject=subject_choice, 
                ai_score=ai_score,
                teacher_score=teacher_score,
                max_m=eval_data['max_score'], 
                grade=eval_data['grade']
            )
            st.success(f"Finalized record for {detected_name} stored.")

    # TEACHER DASHBOARD
    st.divider()
    st.header("📋 Student Record History & Calibration Log")
    df = db.get_all_results_df()
    if not df.empty:
        # Highlight variance to see how accurate the AI is
        st.dataframe(df.style.highlight_max(axis=0, subset=['score_variance'], color='#3d1111'), use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Marksheet", csv, "marksheet_with_calibration.csv", "text/csv")
    else:
        st.info("No records available.")

if __name__ == "__main__":
    main()