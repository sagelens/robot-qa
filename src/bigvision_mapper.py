
import json
import logging
import os
import random
import shutil
import sys

import click
import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

CWD = os.getcwd()

if f"{CWD}/big_vision" not in sys.path:
    sys.path.append(f"{CWD}/big_vision")

from big_vision.pp.proj.paligemma.segmentation import (
    encode_to_codebook_indices,
    get_checkpoint,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)

CHECKPOINT = get_checkpoint(model="oi")

def get_file_names(data_path: str, file_name: str) -> list:
    with open(os.path.join(data_path, file_name), "r") as file:
        return file.read().splitlines()


def reduce_contours(contours, epsilon: float):
    """Reduce the number of points in the contours"""
    approximated_contours = tuple()
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon * perimeter, closed=True)
        approximated_contours += (approx,)
    return approximated_contours


def get_bounding_box(contour):
    x1, y1, w, h = cv2.boundingRect(contour)
    x2, y2 = x1 + w, y1 + h
    return x1, y1, x2, y2


def get_contours_coordinates(ccontours) -> dict:
    reshaped_cnts = [cnt.reshape(len(cnt), 2) for cnt in ccontours]

    contours_coords = dict()
    for n, contour in enumerate(reshaped_cnts):
        flatten_cnt = contour.flatten()
        xvals = [flatten_cnt[x] for x in range(0, len(flatten_cnt), 2)]  # even=x
        yvals = [flatten_cnt[y] for y in range(1, len(flatten_cnt), 2)]  # odd=y
        contours_coords[n] = (xvals, yvals)
    return contours_coords


def plot_image_and_contours(image, contour, points=None):
    cnt_dict = get_contours_coordinates(contour)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    for _, (x, y) in cnt_dict.items():
        ax.plot(x, y, "r-")
    if points is not None:
        for (xp, yp) in points:
            ax.plot(xp, yp, "bo")
    fig.canvas.draw()
    fig.canvas.tostring_argb()
    plt.show()


def format_bbox(y1, x1, y2, x2, h: int, w: int, bbox_tokens: tf.Tensor) -> tf.Tensor:
    bbox = np.array([y1, x1, y2, x2]) / np.array([h, w, h, w])
    binned_loc = tf.cast(tf.round(bbox * 1023), tf.int32)
    # binned_loc = tf.cast(tf.round(bbox * 1), tf.int32)
    binned_loc = tf.clip_by_value(binned_loc, 0, 1023)
    loc_string = tf.strings.reduce_join(tf.gather(bbox_tokens, binned_loc))
    return loc_string


def get_mask_from_contour(h: int, w: int, cnt: np.ndarray) -> np.ndarray:
    new_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    cv2.drawContours(
        new_mask,
        [cnt],
        contourIdx=0,
        color=255,
        thickness=cv2.FILLED,
    )
    # convert to bool
    new_mask = new_mask.astype(bool).copy()
    return new_mask


def format_mask(boolean_mask: np.ndarray, y1, x1, y2, x2, segment_tokens: tf.Tensor):
    tensor_mask = tf.convert_to_tensor(boolean_mask.astype(np.uint8), dtype=tf.uint8)
    yy1 = tf.cast(tf.round(y1), tf.int32)
    xx1 = tf.cast(tf.round(x1), tf.int32)
    yy2 = tf.cast(tf.round(y2), tf.int32)
    xx2 = tf.cast(tf.round(x2), tf.int32)

    tensor_mask = tf.image.resize(
        tensor_mask[None, yy1:yy2, xx1:xx2, None],
        [64, 64],
        method="bilinear",
        antialias=True,
    )
    
    mask_indices = encode_to_codebook_indices(CHECKPOINT, tensor_mask)[0]
    mask_string = tf.strings.reduce_join(tf.gather(segment_tokens, mask_indices))
    return mask_string


def create_output_for_paligemma(
    mask,
    bboxes,
    mask_name: str,
    threshold: int,
    epsilon: float,
    cclass: str,
    prefix: str,
    npoints: int,
) -> dict:
    """Given an image, it creates a dict with the output for paligemma.
    IMPORTANT: This function assumes the same filename for both images and masks."""

    im_height, im_width = mask.shape

    if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
        final_output = {"image": mask_name, "prefix": prefix, "suffix": " "}

    else:
        # make the mask binary
        _, mask_binary = cv2.threshold(
            mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )

        # Define the tokens for the output
        loc_tokens = tf.constant(["<loc%04d>" % i for i in range(1024)])
        seg_tokens = tf.constant(["<seg%03d>" % i for i in range(128)])

        paligemma_output = []
        for box in bboxes:

            y1, x1, y2, x2 = box

            bbox_loc_string = format_bbox(
                y1, x1, y2, x2, im_height, im_width, loc_tokens
            )

            bool_mask = mask_binary.astype(bool).copy()

            mask_loc_string = format_mask(bool_mask, y1, x1, y2, x2, seg_tokens)

            suffix = tf.strings.join([bbox_loc_string, mask_loc_string])

            paligemma_output.append(f"{suffix.numpy().decode('utf-8')} {cclass}")

        paligemma_output = " ; ".join(paligemma_output)

        final_output = {
            "image": mask_name,
            "prefix": prefix,
            "suffix": paligemma_output,
        }

    return final_output