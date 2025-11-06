#!/bin/bash

set -e  # exit on error

CONDA_ENV_NAME="vlm"    # <-- change this to your conda env name
SAM_MODEL_PATH="sam_vit_h.pth"
VAE_MODEL_PATH="vae-oid.npz"

echo "Activating conda environment: $CONDA_ENV_NAME"

if [ -z "$CONDA_EXE" ]; then
    echo "Conda not found. Please ensure conda is installed and on PATH."
    exit 1
fi

eval "$($(dirname $CONDA_EXE)/../bin/conda shell.bash hook)"

# Activate the environment
conda activate "$CONDA_ENV_NAME"

echo "Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"

echo "Downloading model weight files..."
echo "============================================="

# Download SAM ViT-H model
if [ ! -f "$SAM_MODEL_PATH" ]; then
    wget -O "$SAM_MODEL_PATH" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
else
    echo "$SAM_MODEL_PATH already exists, skipping download."
fi

# Download VAE model
if [ ! -f "$VAE_MODEL_PATH" ]; then
    wget -O "$VAE_MODEL_PATH" "https://huggingface.co/spaces/big-vision/paligemma/resolve/main/vae-oid.npz"
else
    echo "$VAE_MODEL_PATH already exists, skipping download."
fi

echo "Model weights ready."

echo "Downloading and saving PaliGemma model..."

python3 <<'PYCODE'
import torch
import os 

CWD = os.cwd()
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

model_id = "google/paligemma2-3b-mix-224"

model_save_path = f"{CWD}/paligemma-mix-local"

print(f"Loading model: {model_id}")
processor = AutoProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Saving model and processor to {model_save_path}")
model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)

print("Paligemma Model and processor saved successfully.")
PYCODE

echo "All models setup completed successfully!"