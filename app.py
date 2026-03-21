# Entry point for Hugging Face Spaces
# HF Spaces looks for app.py in the root by default

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "ui/streamlit_app.py",
    "--server.port=7860",
    "--server.address=0.0.0.0"
])