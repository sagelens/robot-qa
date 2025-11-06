import re
import numpy as np

def parse_boxes_from_suffix(suffix):
    """
    Parse all bounding boxes from a suffix string containing multiple <loc####> groups.
    Each group may represent a crack (4 locs per crack).
    Returns: list of (y1, x1, y2, x2)
    """
    boxes = []

    # Split on semicolons or the word 'crack' to separate crack instances
    parts = re.split(r';|\bcrack\b', suffix)

    for part in parts:
        # Extract all <loc####> tokens from this part
        locs = re.findall(r"<loc(\d+)>", part)
        if not locs:
            continue
        
        locs = np.array([int(l) for l in locs], dtype=np.int32)

        # If there are exactly 4 locs → one box (y1, x1, y2, x2)
        if len(locs) == 4:
            boxes.append(tuple(locs))
        # If more than 4 locs appear, assume they describe multiple boxes in sequence
        elif len(locs) % 4 == 0:
            for i in range(0, len(locs), 4):
                boxes.append(tuple(locs[i:i+4]))
        else:
            # Handle malformed data
            print(f"⚠️ Unexpected number of <loc> tokens ({len(locs)}) in: {part[:80]}")

    return boxes

def pad_boxes(gt_boxes, pred_boxes):
    """
    Pads gt_boxes and pred_boxes to the same length by adding (0,0,0,0).
    Returns padded_gt_boxes, padded_pred_boxes
    """
    max_len = max(len(gt_boxes), len(pred_boxes))
    pad_box = (0, 0, 0, 0)

    gt_padded = gt_boxes + [pad_box] * (max_len - len(gt_boxes))
    pred_padded = pred_boxes + [pad_box] * (max_len - len(pred_boxes))

    return gt_padded, pred_padded