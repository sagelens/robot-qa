# pip install segment-anything pillow pydensecrf
import numpy as np
from PIL import Image, ImageDraw  # Import PIL.Image and ImageDraw
from segment_anything import sam_model_registry, SamPredictor
import os 

CWD = os.getcwd()
sam = sam_model_registry["vit_h"](checkpoint="{CWD}/sam_vit_h.pth")
sam.to(device="cuda")
pred = SamPredictor(sam)

def get_mask(image_path, bbox):
    pil_image = Image.open(image_path)
    pil_image_rgb = pil_image.convert("RGB")
    img = np.array(pil_image_rgb)
    original_width = 1024
    original_height = 1024
    y1_orig, x1_orig, y2_orig, x2_orig = bbox
    new_width, new_height = pil_image_rgb.size
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    x1 = int(round(x1_orig * x_scale))
    y1 = int(round(y1_orig * y_scale))
    x2 = int(round(x2_orig * x_scale))
    y2 = int(round(y2_orig * y_scale))
    roi_box = np.array([x1, y1, x2, y2], dtype=np.int32)
    pred.set_image(img)
    masks, scores, _ = pred.predict(box=roi_box, multimask_output=True)
    best_mask_boolean = masks[np.argmax(scores)]
    mask_to_save = (best_mask_boolean * 255).astype(np.uint8)
    # pil_mask_image = Image.fromarray(mask_to_save, mode='L')
    # pil_mask_image.save(mask_save_path)
    return mask_to_save, [y1, x1, y2, x2]
