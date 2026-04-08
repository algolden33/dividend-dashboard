import subprocess
import sys
import webbrowser
from pathlib import Path

APP = Path(__file__).parent / "app.py"
URL = "http://localhost:8501"

if __name__ == "__main__":
    webbrowser.open(URL)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP)])
