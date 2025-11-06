import json
import os  
import re 
import os, sys
sys.path.append(os.getcwd())
import numpy as np 
import cv2
from pathlib import Path
from PIL import Image
import shutil 
from tqdm import tqdm 

from src.run_sam import get_mask
from src.bigvision_mapper import create_output_for_paligemma
from src.infer_paligemma_best_ckpt import load_model, get_paligemma_tokens
from src.tok_to_mask import paligemma_postprocess
FULL_DATASET_DIR = '{CWD}/all_data/'

SPLITS = ['train', 'valid']

CWD = os.getcwd()

base_id = f"{CWD}/paligemma-mix-local"
adapter_base_dir = f"{CWD}/pg_seg_cracks_lora/"
ckpt_dir = "checkpoint-50"

PROCESSOR, MODEL = load_model(base_id, adapter_base_dir, ckpt_dir)

ANNOTATION_PATH_TEMPLATE = "_annotations.{SPLIT}.jsonl"
PRED_ANNOTATION_PATH_TEMPLATE = "_annotations.predicted.{SPLIT}.jsonl"

# train_annots = ANNOTATION_PATH_TEMPLATE.format(SPLIT='train')

for SPLIT in SPLITS:
    print(f'Processing {SPLIT}')
    annots = os.path.join(FULL_DATASET_DIR, ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
    with open(annots, 'r') as file:
        all_lines = file.readlines()
        predicted_list = []
        for line in tqdm(all_lines, leave = False):
            if line.strip(): 
                data = json.loads(line.strip())
                image_filename = data['image']
                image_path = os.path.join(FULL_DATASET_DIR, 'images/', image_filename)
                prefix = data['prefix']
                suffix = data['suffix']
                predicted_suffix = get_paligemma_tokens(image_path, prefix, PROCESSOR, MODEL)
                predicted_suffix = predicted_suffix.replace("<eos>", "")
                predicted_annot = data 
                predicted_annot['suffix'] = predicted_suffix
                image = Image.open(image_path) 
                overlay, mask = paligemma_postprocess(
                    decoded=predicted_suffix,
                    image=image,
                )
                if mask is not None:
                    mask_pred_filename = os.path.join(FULL_DATASET_DIR, 'predicted_masks/', f"{Path(image_filename).stem}_mask_{prefix}.png")
                    mask.save(mask_pred_filename)
                    predicted_list.append(json.dumps(predicted_annot))

        full_out_path = os.path.join(FULL_DATASET_DIR, PRED_ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
        with open(full_out_path, "a", encoding="utf-8") as file:
            for item in predicted_list:
                json.dump(item, file)
                file.write("\n")