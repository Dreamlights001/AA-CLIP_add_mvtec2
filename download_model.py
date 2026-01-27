#!/usr/bin/env python3
"""
Download script for OpenCLIP models
Downloads ViT-L-14-336px.pt model file to the model directory
"""

import os
import requests
import torch
from tqdm import tqdm

# Model download URL - OpenCLIP official source for OpenAI format models
MODEL_URL = "https://huggingface.co/laion/CLIP-ViT-L-14-336/resolve/main/openai_clip_vit_l_14_336px.pt"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ViT-L-14-336px.pt")

def validate_model():
    """Validate the downloaded model file"""
    try:
        print("Validating model file...")
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        
        # Check if model contains both visual and text components
        has_visual = any("visual." in k for k in state_dict.keys())
        has_text = any(k in ["token_embedding.weight", "positional_embedding", "ln_final.weight"] for k in state_dict.keys())
        
        print(f"Model validation results:")
        print(f"  Contains visual components: {has_visual}")
        print(f"  Contains text components: {has_text}")
        
        if has_visual and has_text:
            print("✓ Model file is valid and contains both visual and text components")
            return True
        else:
            print("✗ Model file is incomplete - missing required components")
            return False
    except Exception as e:
        print(f"Error validating model file: {e}")
        return False

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
        # Validate existing model
        validate_model()
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
    
    # Validate downloaded model
    if validate_model():
        print("This is an OpenAI format CLIP model, compatible with AA-CLIP")
    else:
        print("Warning: Model file may not be compatible with AA-CLIP")
        print("Please try downloading again or use a different model source")

if __name__ == "__main__":
    download_model()
