import cv2
import numpy as np
import glob

def load_and_normalize_mask(path):
    """Load mask as binary (0/1) from 0-255 black/white image."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    return mask

def calculate_seg_metrics(pred_mask, gt_mask):
    """Compute IoU and Dice for 0/1 binary masks."""
    # Ensure same shape
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}")
    
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    p_sum = pred_mask.sum()
    g_sum = gt_mask.sum()

    # IoU
    iou = 1.0 if union == 0 else inter / union
    # Dice
    dice = 1.0 if (p_sum + g_sum) == 0 else 2 * inter / (p_sum + g_sum)

    return float(iou), float(dice)

def calculate_box_metrics(boxA, boxB):
    """
    Compute IoU and Dice between two bounding boxes.
    Format: (y1, x1, y2, x2) = (top, left, bottom, right)
    Returns (iou, dice)
    """

    y1, x1, y2, x2 = boxA
    y1b, x1b, y2b, x2b = boxB

    # --- Intersection box ---
    inter_y1 = max(y1, y1b)
    inter_x1 = max(x1, x1b)
    inter_y2 = min(y2, y2b)
    inter_x2 = min(x2, x2b)

    inter_h = max(0, inter_y2 - inter_y1)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_area = inter_w * inter_h

    # --- Areas ---
    areaA = max(0, (y2 - y1)) * max(0, (x2 - x1))
    areaB = max(0, (y2b - y1b)) * max(0, (x2b - x1b))
    union = areaA + areaB - inter_area

    # --- IoU ---
    iou = inter_area / union if union > 0 else 0.0

    # --- Dice (F1-like) ---
    dice = (2 * inter_area) / (areaA + areaB) if (areaA + areaB) > 0 else 0.0

    return float(iou), float(dice)

