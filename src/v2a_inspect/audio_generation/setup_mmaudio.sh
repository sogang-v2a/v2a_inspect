#!/bin/bash
set -e

echo "Installing ffmpeg..."
apt-get update
apt-get install -y ffmpeg

echo "Cloning MMAudio..."
cd /root
if [ ! -d "MMAudio" ]; then
    git clone https://github.com/hkchengrex/MMAudio.git
fi
cd MMAudio

echo "Installing pip requirements..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install fastapi uvicorn python-multipart huggingface_hub requests

echo "Copying serve script..."
cp /root/serve_mmaudio.py /root/MMAudio/serve_mmaudio.py

echo "Starting MMAudio FastAPI server on port 8080..."
pkill -f serve_mmaudio || true
nohup python serve_mmaudio.py > /root/MMAudio/serve.log 2>&1 &

echo "Deployment finished! MMAudio is starting in the background. It will take a few minutes to download weights on first run."
