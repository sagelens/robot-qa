import json
import os
import sys
import numpy as np 
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.metrics import calculate_seg_metrics, load_and_normalize_mask, calculate_box_metrics  # assuming these exist
from src.utils import parse_boxes_from_suffix, pad_boxes
CWD = os.getcwd()
FULL_DATASET_DIR = f"{CWD}/all_data"
SPLITS = ["train", "valid"]

# Predicted annotation file template
PRED_ANNOTATION_PATH_TEMPLATE = "_annotations.predicted.{SPLIT}.jsonl"
GT_ANNOTATION_PATH_TEMPLATE = "_annotations.{SPLIT}.jsonl"

for SPLIT in SPLITS:
    print(f"\nProcessing {SPLIT}")

    results = {
        "segment cracks": {"seg_iou": [], "seg_dice": [], "box_iou": [], "box_dice": []},
        "segment drywall": {"seg_iou": [], "seg_dice": [], "box_iou": [], "box_dice": []},
    }

    # Paths
    pred_annot_path = os.path.join(FULL_DATASET_DIR, PRED_ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))
    gt_annot_path = os.path.join(FULL_DATASET_DIR, GT_ANNOTATION_PATH_TEMPLATE.format(SPLIT=SPLIT))

    # Load GT annotations into a dict (for matching by image name)
    gt_data = {}
    with open(gt_annot_path, "r") as gt_file:
        for line in gt_file:
            if line.strip():
                item = json.loads(line.strip())
                if isinstance(item, str):
                    item = json.loads(item)
                gt_data[item["image"]] = item

    # Load predicted annotations
    with open(pred_annot_path, "r") as file:
        all_lines = file.readlines()

    for line in tqdm(all_lines, desc=f"Evaluating {SPLIT}", leave=False):
        if not line.strip():
            continue

        data = json.loads(line.strip())
        if isinstance(data, str):
            data = json.loads(data)

        image_filename = data["image"]
        prefix = data["prefix"]
        pred_suffix = data.get("suffix", "")

        # Get GT suffix for reference
        gt_suffix = gt_data.get(image_filename, {}).get("suffix", "")

        # Load masks
        mask_gt_filename = os.path.join(FULL_DATASET_DIR, "masks", f"{Path(image_filename).stem}_mask.png")
        mask_pred_filename = os.path.join(FULL_DATASET_DIR, "predicted_masks", f"{Path(image_filename).stem}_mask_{prefix}.png")

        if not os.path.exists(mask_gt_filename):
            print(f"Missing GT mask: {mask_gt_filename}")
            continue
        if not os.path.exists(mask_pred_filename):
            print(f"Missing Pred mask: {mask_pred_filename}")
            continue

        gt_mask = load_and_normalize_mask(mask_gt_filename)
        pred_mask = load_and_normalize_mask(mask_pred_filename)

        # Calculate metrics
        iou, dice = calculate_seg_metrics(gt_mask, pred_mask)

        if prefix not in results:
            results[prefix] = {"seg_iou": [], "seg_dice": []}

        results[prefix]["seg_iou"].append(iou)
        results[prefix]["seg_dice"].append(dice)

        box_gt = parse_boxes_from_suffix(gt_suffix)
        box_pred = parse_boxes_from_suffix(pred_suffix)
        iou = 0
        dice = 0
        gt_padded, pred_padded = pad_boxes(box_gt, box_pred)
        for (b1, b2) in zip(gt_padded, pred_padded):
            iou_, dice_ = calculate_box_metrics(b1, b2)
            iou += iou_ 
            dice += dice_
        results[prefix]["box_iou"].append(iou)
        results[prefix]["box_dice"].append(dice)


    out_path = os.path.join(FULL_DATASET_DIR, f"{SPLIT}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {out_path}")

    for key, vals in results.items():
        if vals["seg_iou"]:
            print(f"  {key}: mIoU (segmentation)={np.mean(vals['seg_iou']):.4f}, mDice (segmentation)={np.mean(vals['seg_dice']):.4f}")
        if vals["box_iou"]:
            print(f"  {key}: mIoU (bounding box)={np.mean(vals['box_iou']):.4f}, mDice (bounding box)={np.mean(vals['box_dice']):.4f}")
    
    print("\nValidation Consistency Analysis:\n")

    # Combine all per-image IoUs for global stats
    all_seg_ious = []
    for key in results:
        all_seg_ious.extend(results[key]["seg_iou"])
    all_seg_ious = np.array(all_seg_ious)

    if len(all_seg_ious) > 0:
        mean_iou = np.mean(all_seg_ious)
        std_iou = np.std(all_seg_ious)
        min_iou = np.min(all_seg_ious)
        max_iou = np.max(all_seg_ious)
        p10, p50, p90 = np.percentile(all_seg_ious, [10, 50, 90])

        print("mIoU Per-Image Stats:")
        print(f"  Mean:        {mean_iou:.4f}")
        print(f"  Std Dev:     {std_iou:.4f}  ← Lower is better")
        print(f"  Min-Max:     {min_iou:.4f} - {max_iou:.4f}\n")

        print("Per-Prompt:")
        for key in results.keys():
            if results[key]["seg_iou"]:
                mean_p = np.mean(results[key]["seg_iou"])
                std_p = np.std(results[key]["seg_iou"])
                task = key.replace("segment ", "").capitalize()
                print(f"  {task} mIoU:  {mean_p:.4f} ± {std_p:.4f}")
        print()

        print("Percentiles:")
        print(f"  P10:  {p10:.4f}")
        print(f"  P50:  {p50:.4f}")
        print(f"  P90:  {p90:.4f}")
        print("\n" + "=" * 60 + "\n")
    else:
        print("No valid mIoU data to summarize.\n")