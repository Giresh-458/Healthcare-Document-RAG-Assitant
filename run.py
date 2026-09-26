import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import subprocess
import sys
import time

def main():
    print("[INFO] Starting Healthcare Document RAG Assistant...")
    
    # 1. Start FastAPI Backend
    print("[INFO] Starting FastAPI backend on http://0.0.0.0:8000...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    time.sleep(2)  # Give backend a moment to initialize
    
    # 2. Start Streamlit Frontend
    streamlit_port = os.environ.get("PORT", "8501")
    print(f"[INFO] Starting Streamlit frontend on port {streamlit_port}...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", streamlit_port, "--server.address", "0.0.0.0"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Terminating backend and frontend processes...")
        backend.terminate()
        frontend.terminate()

if __name__ == "__main__":
    main()
