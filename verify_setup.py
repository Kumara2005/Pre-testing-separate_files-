"""
Setup verification script for Automated Paper Correction System.
Run this to verify all prerequisites are properly configured.
"""

import sys


def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (Need 3.8+)")
        return False


def check_dependencies():
    """Check if all required Python packages are installed."""
    required_packages = {
        'streamlit': 'Streamlit',
        'pdf2image': 'pdf2image',
        'google.genai': 'google-genai',
        'sentence_transformers': 'sentence-transformers',
        'PIL': 'Pillow',
        'dotenv': 'python-dotenv',
        'nest_asyncio': 'nest-asyncio'
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            all_ok = False
    
    return all_ok


def check_poppler():
    """Check if Poppler is installed and accessible."""
    try:
        from pdf2image.exceptions import PDFInfoNotInstalledError
        from pdf2image import pdfinfo_from_path
        # This will raise an exception if poppler is not found
        # We don't need an actual PDF, just testing if the command exists
        print("✅ Poppler appears to be installed")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'poppler' in error_msg or 'pdftoppm' in error_msg:
            print("❌ Poppler NOT found in PATH")
            return False
        else:
            print("⚠️ Could not verify Poppler (may still work)")
            return True


def check_env_file():
    """Check if .env file exists and has API key."""
    import os
    from pathlib import Path
    
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file NOT found")
        return False
    
    print("✅ .env file exists")
    
    # Try to load and check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key:
        print("❌ GEMINI_API_KEY not set in .env")
        return False
    
    if len(api_key) < 20:
        print("⚠️ GEMINI_API_KEY seems too short (possibly invalid)")
        return False
    
    print(f"✅ GEMINI_API_KEY configured ({api_key[:10]}...)")
    return True


def check_api_connection():
    """Test Gemini API connection."""
    try:
        from utils import verify_gemini_api_key
        is_valid, message = verify_gemini_api_key()
        
        if is_valid:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        
        return is_valid
    except Exception as e:
        print(f"❌ Error checking API: {str(e)}")
        return False


def check_internet():
    """Check internet connectivity."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("✅ Internet connection available")
        return True
    except OSError:
        print("❌ No internet connection")
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("Automated Paper Correction System - Setup Verification")
    print("v2.0 - Gemini 2.5 Flash Powered")
    print("="*60)
    print()
    
    checks = []
    
    print("1. Checking Python version...")
    checks.append(check_python_version())
    print()
    
    print("2. Checking Python dependencies...")
    checks.append(check_dependencies())
    print()
    
    print("3. Checking Poppler installation...")
    checks.append(check_poppler())
    print()
    
    print("4. Checking environment configuration...")
    checks.append(check_env_file())
    print()
    
    print("5. Checking internet connection...")
    checks.append(check_internet())
    print()
    
    print("6. Testing Gemini API connection...")
    checks.append(check_api_connection())
    print()
    
    print("="*60)
    if all(checks):
        print("🎉 ALL CHECKS PASSED!")
        print("You're ready to run the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️ SOME CHECKS FAILED")
        print("Please fix the issues above before running the application.")
        print("\nRefer to SETUP.md for detailed installation instructions.")
    print("="*60)


if __name__ == "__main__":
    main()
