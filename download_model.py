#!/usr/bin/env python3
"""
Download script for OpenCLIP models
Downloads ViT-L-14-336px.pt model file to the model directory
"""

import os
import requests
from tqdm import tqdm

# Model download URL
MODEL_URL = "https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ViT-L-14-336px.pt")

def download_model():
    """Download the ViT-L-14-336px.pt model file"""
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"Model file already exists: {MODEL_PATH}")
        return
    
    print(f"Downloading model from {MODEL_URL}")
    print(f"Saving to {MODEL_PATH}")
    
    # Download with progress bar
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(MODEL_PATH, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    print(f"Download completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    download_model()
