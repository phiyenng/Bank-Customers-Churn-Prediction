"""
Run the Streamlit Bank Customer Churn Prediction App
===================================================

This script launches the Streamlit web application.

Usage:
    python run_app.py

Or directly with streamlit:
    streamlit run app.py
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
    
    # Launch the app
    print("🚀 Launching Bank Customer Churn Prediction App...")
    print("📱 The app will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*60)
    
    # Run streamlit
    os.system("streamlit run app.py --server.port 8501 --server.address localhost")

if __name__ == "__main__":
    main()
