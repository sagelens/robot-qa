#!/usr/bin/env bash
set -e  # Exit immediately on error

# Step 1: Create a new conda environment
echo "Creating conda environment: vlm (Python 3.10)..."
conda create -n vlm python=3.10 -y

# Step 2: Activate environment in the current shell
# Conda activate won't work in non-interactive shells directly, so use conda's 'source' init.
echo "Activating vlm..."
eval "$(conda shell.bash hook)"
conda activate vlm

# Step 4: Install TensorFlow with CUDA support
echo "Installing TensorFlow (GPU-enabled)..."
pip install "tensorflow[and-cuda]"

# Step 5: Install PyTorch with matching CUDA version
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Environment setup complete!"
