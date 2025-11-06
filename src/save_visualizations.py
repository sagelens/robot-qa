import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.utils import parse_boxes_from_suffix, pad_boxes   # assumed to exist
from src.metrics import load_and_normalize_mask

CWD = os.getcwd()
FULL_DATASET_DIR = f"{CWD}/all_data"
OUT_DIR = os.path.join(CWD, "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FILES = [
    "3872_jpg.rf.f4ec798bfc519354376209d0a8d6e3b1.jpg",
    "IMG_8209_JPG_jpg.rf.8fbaa3c995fac30818b643910225b928.jpg",
    "WP6S2G7E75FFPFLA3RLQBAQGWQ_jpg.rf.59e85935f19d5885a4e5dd981eb243fe.jpg",
    "2000x1500_5_resized_jpg.rf.6423c25a172441fa1f66e4e927fe68a9.jpg",
    "IMG_8223_JPG_jpg.rf.121531b6267b5a04f871202c0dbc43c6.jpg",
    "cracking0102204_jpg.rf.0d64df9e69ef678b6185988b7e030387.jpg",
]

PRED_ANNOTATION_PATH_TEMPLATE = "_annotations.predicted.{SPLIT}.jsonl"
GT_ANNOTATION_PATH_TEMPLATE = "_annotations.{SPLIT}.jsonl"

def scale_boxes(boxes, img_shape, base=1023):
    """Scale [y1, x1, y2, x2] boxes from 0–base range to actual image dimensions."""
    if not boxes:
        return boxes
    h, w = img_shape[:2]
    scaled = []
    for (y1, x1, y2, x2) in boxes:
        y1 = (y1 / base) * h
        y2 = (y2 / base) * h
        x1 = (x1 / base) * w
        x2 = (x2 / base) * w
        scaled.append([y1, x1, y2, x2])
    return scaled

def draw_boxes(ax, boxes, color="lime", label=None):
    """Draw bounding boxes on matplotlib axis."""
    for box in boxes:
        y1, x1, y2, x2 = box  # correct order
        ax.plot([x1, x2, x2, x1, x1],
                [y1, y1, y2, y2, y1],
                color=color, linewidth=2)
    if label:
        ax.set_title(label, fontsize=9)

def overlay_mask(ax, img, mask, alpha=0.5, cmap="Reds", title=""):
    ax.imshow(img)
    ax.imshow(mask, cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)

def make_canvas(image_path, prefix, gt_boxes, pred_boxes, gt_mask, pred_mask, out_path):
    """Compact 3-panel layout (Original | GT Row | Pred Row) with minimal whitespace."""
    img = np.array(Image.open(image_path).convert("RGB"))

    # Compact layout (shorter height, small width padding)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(
        f"{Path(image_path).name} | {prefix}",
        fontsize=12,
        fontweight="bold",
        y=0.97
    )

    grid = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

    ax_orig = fig.add_subplot(grid[0, :])
    ax_orig.imshow(img)
    ax_orig.axis("off")
    ax_orig.set_title("Original Image", fontsize=11, pad=1)

    ax_gt_box = fig.add_subplot(grid[1, 0])
    ax_gt_box.imshow(img)
    draw_boxes(ax_gt_box, gt_boxes, color="green")
    ax_gt_box.axis("off")
    ax_gt_box.set_title("GT Boxes (Dataset annotations)", fontsize=10, pad=1)

    ax_gt_seg = fig.add_subplot(grid[1, 1])
    ax_gt_seg.imshow(img)
    ax_gt_seg.imshow(gt_mask, cmap="Greens", alpha=0.5)
    ax_gt_seg.axis("off")
    ax_gt_seg.set_title("GT Segmentation Mask (Psuedo Labelled via Segment Anything)", fontsize=10, pad=1)

    ax_pred_box = fig.add_subplot(grid[2, 0])
    ax_pred_box.imshow(img)
    draw_boxes(ax_pred_box, pred_boxes, color="yellow")
    ax_pred_box.axis("off")
    ax_pred_box.set_title("Predicted Boxes", fontsize=10, pad=1)

    ax_pred_seg = fig.add_subplot(grid[2, 1])
    ax_pred_seg.imshow(img)
    ax_pred_seg.imshow(pred_mask, cmap="Reds", alpha=0.5)
    ax_pred_seg.axis("off")
    ax_pred_seg.set_title("Predicted Segmentation Mask (Predicted via Fine Tuned Paligemma)", fontsize=10, pad=1)

    plt.subplots_adjust(
        top=0.93, bottom=0.03, left=0.03, right=0.97, hspace=0.15, wspace=0.02
    )

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)





for SPLIT in ["train", "valid"]:
    pred_annot_path = os.path.join(FULL_DATASET_DIR, PRED_ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
    gt_annot_path = os.path.join(FULL_DATASET_DIR, GT_ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))

    if not os.path.exists(pred_annot_path):
        continue

    # Load GT annotations
    gt_data = {}
    if os.path.exists(gt_annot_path):
        with open(gt_annot_path, "r") as gt_file:
            for line in gt_file:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                if isinstance(item, str):
                    item = json.loads(item)
                gt_data[item["image"]] = item

    # Load predicted annotations
    with open(pred_annot_path, "r") as f:
        lines = f.readlines()

    print(f"\nProcessing {SPLIT}...")

    for line in tqdm(lines, desc=f"Scanning {SPLIT}"):
        if not line.strip():
            continue

        data = json.loads(line.strip())
        if isinstance(data, str):
            data = json.loads(data)

        image_filename = data["image"]
        prefix = data["prefix"]
        pred_suffix = data.get("suffix", "")

        if not any(t in image_filename for t in TARGET_FILES):
            continue

        gt_suffix = gt_data.get(image_filename, {}).get("suffix", "")

        img_path = os.path.join(FULL_DATASET_DIR, "images", image_filename)
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        # Masks
        mask_gt_filename = os.path.join(FULL_DATASET_DIR, "masks", f"{Path(image_filename).stem}_mask.png")
        mask_pred_filename = os.path.join(FULL_DATASET_DIR, "predicted_masks", f"{Path(image_filename).stem}_mask_{prefix}.png")

        if not (os.path.exists(mask_gt_filename) and os.path.exists(mask_pred_filename)):
            print(f"Missing masks for {image_filename}")
            continue

        gt_mask = load_and_normalize_mask(mask_gt_filename)
        pred_mask = load_and_normalize_mask(mask_pred_filename)

        # Boxes
        box_gt = parse_boxes_from_suffix(gt_suffix)
        box_pred = parse_boxes_from_suffix(pred_suffix)
        gt_padded, pred_padded = pad_boxes(box_gt, box_pred)

        img = np.array(Image.open(img_path))
        gt_scaled = scale_boxes(gt_padded, img.shape)
        pred_scaled = scale_boxes(pred_padded, img.shape)

        prefix_short = prefix.replace("segment ", "")
        out_name = f"{Path(image_filename).stem}_{prefix_short}_canvas.jpg"
        out_path = os.path.join(OUT_DIR, out_name)

        make_canvas(
            image_path=img_path,
            prefix=prefix,
            gt_boxes=gt_scaled,
            pred_boxes=pred_scaled,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            out_path=out_path
        )

        print(f"Saved canvas: {out_path}")