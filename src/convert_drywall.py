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

CWD = os.getcwd()

CRACK_DATASET_DIR = f'{CWD}/drywall/'
OUT_DATA_DIR = f'{CWD}/all_data/'

SPLITS = ['train', 'valid']

ANNOTATION_PATH_TEMPLATE = "_annotations.{SPLIT}.jsonl"

# train_annots = ANNOTATION_PATH_TEMPLATE.format(SPLIT='train')

for SPLIT in SPLITS:
    print(f'Processing {SPLIT}')
    annots = os.path.join(CRACK_DATASET_DIR, ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
    with open(annots, 'r') as file:
        all_lines = file.readlines()
        paligemma_list = []
        for line in tqdm(all_lines, leave = False):
            if line.strip(): 
                data = json.loads(line.strip())
                image_filename = data['image']
                image_path = os.path.join(CRACK_DATASET_DIR, image_filename)
                prefix = 'segment drywall'
                suffix = data['suffix']
                suffix_list = suffix.split(';')
                if len(suffix_list) >= 1:
                    cur_mask = None
                    bboxes = []
                    for suffix_i in suffix_list:
                        bbox_string = suffix_i.strip().split(' ')[0]
                        matches = re.findall(r'<loc(\d+)>', bbox_string)
                        coordinates = [int(num) for num in matches]
                        if not coordinates or len(coordinates) != 4:
                            print(f"Warning: Invalid coordinates found for {image_path}. Skipping.")
                            continue
                        
                        y1, x1, y2, x2 = coordinates                        
                        mask_i, box = get_mask(image_path, coordinates)
                        
                        if mask_i.size == 0:
                            print(f"Warning: get_mask() returned empty mask for {image_path} with coords {coordinates}. Skipping.")
                            continue  # Go to the next suffix_i
                        bboxes.append(box)
                        if cur_mask is not None:
                            cur_mask = cur_mask | mask_i
                        else:
                            cur_mask = mask_i
                    
                    if cur_mask is not None:
                        try:
                            paligemma_str = create_output_for_paligemma(
                                mask = cur_mask,
                                bboxes = bboxes,
                                mask_name = f"{image_filename}",
                                threshold = 150,
                                epsilon = 1e-3,
                                cclass = 'drywall',
                                prefix = 'segment drywall',
                                npoints = 8,
                            )
                            paligemma_list.append(paligemma_str)
                            mask_gt_filename = os.path.join(OUT_DATA_DIR, 'masks/', f"{Path(image_filename).stem}_mask.png")
                            img = Image.fromarray(cur_mask)
                            img.save(mask_gt_filename)
                            shutil.copy2(image_path, f"{OUT_DATA_DIR}/images/{image_filename}")
                        except:
                            print(cur_mask.shape)
                            pass
                

        full_out_path = os.path.join(OUT_DATA_DIR, ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
        with open(full_out_path, "a", encoding="utf-8") as file:
            for item in paligemma_list:
                json.dump(item, file)
                file.write("\n")