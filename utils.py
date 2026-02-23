"""
Utility functions for file handling and environment configuration.
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Retrieve environment variable with optional default value.
    
    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not found
        
    Returns:
        Value of the environment variable or default
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found and no default provided")
    return value


def save_uploaded_file(uploaded_file, destination_dir: Optional[str] = None) -> str:
    """
    Save an uploaded file to a temporary or specified directory.
    
    Args:
        uploaded_file: File object from Streamlit uploader
        destination_dir: Optional directory path to save the file
        
    Returns:
        Path to the saved file
    """
    if destination_dir is None:
        destination_dir = tempfile.gettempdir()
    
    os.makedirs(destination_dir, exist_ok=True)
    file_path = os.path.join(destination_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def validate_file_extension(file_path: str, allowed_extensions: list) -> bool:
    """
    Validate if a file has an allowed extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.png'])
        
    Returns:
        True if extension is allowed, False otherwise
    """
    file_extension = Path(file_path).suffix.lower()
    return file_extension in [ext.lower() for ext in allowed_extensions]


def save_json(data: Dict[Any, Any], file_path: str) -> None:
    """
    Save dictionary data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path where JSON file should be saved
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Dict[Any, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def cleanup_temp_files(*file_paths: str) -> None:
    """
    Remove temporary files.
    
    Args:
        *file_paths: Variable number of file paths to remove
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove file {file_path}: {e}")


def format_score(score: float, total: float = 100.0) -> str:
    """
    Format a score as a readable string.
    
    Args:
        score: The score value
        total: Total possible score
        
    Returns:
        Formatted score string
    """
    return f"{score:.2f}/{total:.2f}"


def get_api_key(api_name: str = "GEMINI_API_KEY") -> str:
    """
    Get API key from environment variables.
    
    Args:
        api_name: Name of the API key environment variable
        
    Returns:
        API key string
    """
    return get_env_variable(api_name, "")


def verify_gemini_api_key() -> tuple[bool, str]:
    """
    Verify that Gemini API key is configured and valid.
    
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not api_key:
            return False, "GEMINI_API_KEY not found in environment variables. Please set it in .env file or environment."
        
        if len(api_key) < 20:
            return False, "GEMINI_API_KEY appears to be invalid (too short)."
        
        # Basic format validation for Google API keys
        if not api_key.startswith("AIza"):
            return False, "GEMINI_API_KEY format appears invalid (should start with 'AIza')."
        
        # Skip actual API client initialization during verification to avoid SSL/network issues
        # The actual API call will validate the key when the system is used
        return True, f"Gemini API key found (format valid, length: {len(api_key)} chars)."
    
    except Exception as e:
        return False, f"Error verifying Gemini API key: {str(e)}"


def check_api_prerequisites() -> tuple[bool, List[str]]:
    """
    Check all prerequisites for API operations.
    
    Returns:
        Tuple of (all_ok, list_of_issues)
    """
    issues = []
    
    # Check Gemini API key
    is_valid, message = verify_gemini_api_key()
    if not is_valid:
        issues.append(f"❌ {message}")
    else:
        issues.append(f"✅ {message}")
    
    # Check internet connectivity (simple check)
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        issues.append("✅ Internet connection available.")
    except OSError:
        issues.append("⚠️ Internet connection may be unavailable.")
    
    all_ok = is_valid  # Main requirement is the API key
    return all_ok, issues
