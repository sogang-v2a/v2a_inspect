#!/usr/bin/env python
"""
V2A Inspect 앱을 ngrok과 함께 실행 (port 8503 기본값).

사용법:
    python run.py
    python run.py --port 8503 --authtoken YOUR_TOKEN

ngrok 인증 (최초 1회):
    ngrok config add-authtoken YOUR_AUTH_TOKEN
"""

import subprocess
import sys
import time
import argparse
import atexit
import signal
import socket
import os
from pathlib import Path
from dotenv import load_dotenv


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def cleanup(streamlit_proc, ngrok_module=None):
    if streamlit_proc and streamlit_proc.poll() is None:
        print("Terminating Streamlit...")
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing Streamlit...")
            streamlit_proc.kill()
    if ngrok_module:
        try:
            ngrok_module.kill()
        except Exception:
            pass
    print("Cleanup complete.")


def main():
    # Load env from project root's .env.secure
    env_path = Path(__file__).parent / ".env.secure"
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(description="Run V2A Inspect with ngrok")
    parser.add_argument("--port", type=int, default=8503, help="Streamlit port (default: 8503)")
    parser.add_argument("--authtoken", type=str, default=None, help="ngrok auth token (optional)")
    args = parser.parse_args()

    if is_port_in_use(args.port):
        print(f"Error: Port {args.port} is already in use.")
        print(f"Run: lsof -i :{args.port}  to see what's using it")
        sys.exit(1)

    # Setup ngrok auth token
    ngrok = None
    authtoken = args.authtoken or os.getenv("NGROK_AUTHTOKEN")
    if authtoken:
        from pyngrok import ngrok as ngrok_module
        ngrok = ngrok_module
        ngrok.set_auth_token(authtoken)
        print("ngrok auth token set!")

    # Start Streamlit — run app.py from the inspect directory
    app_path = Path(__file__).parent / "app.py"
    print(f"Starting V2A Inspect on port {args.port}...")
    streamlit_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--server.address", "localhost",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if ngrok is None:
        from pyngrok import ngrok as ngrok_module
        ngrok = ngrok_module

    atexit.register(cleanup, streamlit_proc, ngrok)

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        cleanup(streamlit_proc, ngrok)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    print("Waiting for Streamlit to start...")
    time.sleep(5)

    if streamlit_proc.poll() is not None:
        print("Error: Streamlit failed to start!")
        _, stderr = streamlit_proc.communicate()
        if stderr:
            print(f"Error output:\n{stderr.decode()}")
        sys.exit(1)

    try:
        public_url = ngrok.connect(args.port)
        print("\n" + "=" * 60)
        print("V2A Inspect is now publicly accessible!")
        print("=" * 60)
        print(f"\nPublic URL: {public_url}")
        print(f"Local URL:  http://localhost:{args.port}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        streamlit_proc.wait()

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTip: ngrok 무료 계정은 세션 제한이 있습니다.")
        print("   https://ngrok.com 에서 가입 후 authtoken을 설정하세요:")
        print(f"   python run.py --authtoken YOUR_TOKEN")


if __name__ == "__main__":
    main()
