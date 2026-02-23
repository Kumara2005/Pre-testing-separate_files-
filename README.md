# 📝 Automated Paper Correction System

A powerful, modular Python-based system for automatically correcting student papers using **Gemini 2.5 Flash** for intelligent document processing and AI-powered semantic analysis.

## 🌟 Features

- **Gemini 2.5 Flash Integration**: State-of-the-art document text extraction and semantic analysis
- **Multi-page PDF Support**: Process complete exam scripts and answer keys
- **Async Processing**: Efficient concurrent processing of document pages
- **Intelligent Text Extraction**: No OCR required - Gemini understands document structure
- **Conceptual Accuracy Comparison**: Focus on understanding rather than exact word matching
- **Detailed Feedback Generation**: Human-like explanations for grades
- **Rate Limit Handling**: Automatic backoff and retry logic for API stability
- **Structured Scoring**: JSON-formatted evaluation reports
- **User-Friendly Interface**: Streamlit web application
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
Pre-testing/
├── app.py              # Streamlit web interface
├── pipeline.py         # Main orchestrator
├── extraction.py       # Async text extraction from PDFs
├── compare.py          # Semantic comparison logic
├── evaluation.py       # Grading and scoring system
├── feedback.py         # Feedback generation module
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- **Gemini API Key** (Required) - Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Poppler (for PDF to image conversion)
  - **Windows**: Download from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases/) and add to PATH
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`

### Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Poppler** (see prerequisites above)

4. **Create a `.env` file** (Required):
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the Application

#### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Option 2: Python Script

```python
from pipeline import run_correction_pipeline

results = run_correction_pipeline(
    teacher_file_path="path/to/teacher_key.pdf",
    student_file_path="path/to/student_script.pdf",
    comparison_method="sentence_transformers",  # or "gemini"
    use_ai_feedback=False,  # Set to True for AI-generated feedback
    total_marks=100.0,
    output_dir="results",
    save_results=True
)
```

## 📖 Usage Guide

### Using the Streamlit Interface

1. **Upload Files**:
   - Upload the teacher's answer key (PDF or image)
   - Upload the student's script (PDF or image)

2. **Configure Settings** (in sidebar):
   - Choose comparison method (Sentence Transformers or Gemini)
   - Enable/disable AI feedback
   - Set total marks and pass threshold
   - Add Gemini API key if using AI features

3. **Process**:
   - Click "Process and Evaluate" button
   - Wait for the system to complete analysis

4. **View Results**:
   - See overall score, grade, and status
   - Review page-wise analysis
   - Read detailed feedback
   - Download reports (JSON, text, summary)

## 🔧 Module Details

### `extraction.py`
**Powered by Gemini 2.5 Flash** for intelligent text extraction from PDF and image files.

**Key Features**:
- Converts PDF pages to images using pdf2image
- Sends images to Gemini with structured transcription prompts
- Concurrent page processing for efficiency
- Automatic rate limit handling with exponential backoff
- Maintains document structure and formatting
- Error handling for API failures

### `compare.py`
Performs semantic comparison using Gemini 2.5 Flash for conceptual accuracy analysis.

**Key Features**:
- **Default: Gemini 2.5 Flash** - Conceptual accuracy focus, not exact word matching
- **Fallback: Sentence Transformers** - Available if API issues occur
- Rate limit handling with automatic retries
- Detailed analysis of student understanding

### `evaluation.py`
Calculates scores and generates evaluation reports.

**Features**:
- Configurable grading curves
- Letter grade assignment
- JSON report generation
- Performance metrics

### `feedback.py`
Generates human-like feedback explaining grades.

**Feedback Types**:
- Template-based feedback (fast, no API required)
- AI-generated feedback (detailed, requires Gemini API)
- Page-wise and overall feedback
- Constructive suggestions for improvement

### `pipeline.py`
Orchestrates the entire correction workflow with safety checks.

**Workflow Phases**:
1. **Prerequisites Check**: Verifies API key and connectivity
2. **Extraction**: Extract text using Gemini from both documents
3. **Comparison**: Compare student answers with key using Gemini
4. **Evaluation**: Calculate scores and grades
5. **Feedback**: Generate detailed feedback

### `utils.py`
Utility functions including API verification and rate limit handling.

**Key Functions**:
- `verify_gemini_api_key()`: Validates API configuration
- `check_api_prerequisites()`: Comprehensive system check
- File handling utilities
- JSON operations

### `app.py`
Streamlit web interface with async loop fixes for compatibility.

## 📊 Output Format

### JSON Evaluation Report
```json
{
  "metadata": {
    "evaluation_date": "2026-02-22T...",
    "teacher_file": "answer_key.pdf",
    "student_file": "student_script.pdf"
  },
  "evaluation": {
    "total_score": 85.5,
    "max_score": 100.0,
    "percentage": 85.5,
    "grade": "A",
    "status": "pass",
    "average_similarity": 87.3,
    "page_scores": [...]
  }
}
```

## ⚙️ Configuration Options

### Processing Methods

**Gemini 2.5 Flash** (Default & Recommended):
- Pros: Superior accuracy, intelligent text extraction, conceptual understanding
- Cons: Requires API key, internet connection, usage costs
- Best for: Production use, accurate evaluations, high-stakes exams

**Sentence Transformers** (Fallback):
- Pros: Free, fast, works offline
- Cons: Less accurate, exact matching only
- Best for: Testing, offline scenarios

### Grading Parameters

- `total_marks`: Total marks for the assessment (default: 100)
- `pass_threshold`: Minimum percentage to pass (default: 40%)
- Grading curve can be customized in `evaluation.py`

## 🔑 API Configuration

### Getting Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Add to `.env` file or enter in Streamlit sidebar

**Important**: Keep your API key secure and never commit it to version control.

### Rate Limits

The system includes automatic rate limit handling:
- Exponential backoff on rate limit errors
- Automatic retries (up to 3 attempts)
- Graceful fallback to sentence transformers if needed

## 🛠️ Customization

### Modify Grading Curve
Edit `calculate_page_score()` in [evaluation.py](evaluation.py#L27)

### Change Feedback Templates
Edit `_generate_template_feedback()` in [feedback.py](feedback.py#L52)

### Adjust Similarity Thresholds
Modify scoring multipliers in [evaluation.py](evaluation.py#L38-L50)

## 📝 Requirements

- Python 3.8+
- Gemini API Key (Required)
- Poppler (for PDF processing)
- 2GB+ RAM (for Sentence Transformers fallback)
- Internet connection (for Gemini API)

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found" error
- Create a `.env` file in the project root
- Add `GEMINI_API_KEY=your_key_here`
- Or enter the key in Streamlit sidebar

### "Poppler not found" error
- Install Poppler (see installation instructions)
- Add Poppler's bin directory to system PATH
- Restart terminal/IDE after installation

### Rate limit errors
- The system has automatic retry logic
- If persistent, wait a few minutes
- Consider reducing concurrent workers in extraction.py
- Check your API quota at [Google AI Studio](https://makersuite.google.com)

### Async loop errors in Streamlit
- The app includes nest_asyncio to fix this
- If issues persist, restart the Streamlit server
- Update to latest version: `pip install --upgrade streamlit`

### Low accuracy in results
- Ensure documents are clear and legible
- Check that Gemini successfully extracted text
- Review the extracted_data.json file
- Consider adjusting grading curve parameters

### API errors or timeouts
- Verify internet connection
- Check API key validity
- System will automatically fallback to sentence transformers
- Review error messages in console output

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Support

For issues or questions, please check the troubleshooting section above or review the module documentation in the source code.

## 🎯 Future Enhancements

- [ ] Support for handwritten text recognition via Gemini Vision
- [ ] Multiple language support
- [ ] Question-wise grading (not just page-wise)
- [ ] Batch processing of multiple students
- [ ] Export to Excel/CSV
- [ ] Teacher review interface
- [ ] Historical performance tracking
- [ ] Plagiarism detection
- [ ] Advanced analytics dashboard

---

**v2.0 - Powered by Gemini 2.5 Flash**  
Built with ❤️ using Python, Streamlit, and Google AI
