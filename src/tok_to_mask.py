from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# pip install jax jaxlib flax numpy pillow transformers torch regex
# Place 'vae-oid.npz' in the working directory (from the official reference assets)

import re
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jax
import jax.numpy as jnp
import flax.linen as nn

# ---------- Parsing <loc> and <seg> ----------
# Matches: optional text, 4 locs, optional 16 segs, optional label, with optional '; ' separators
_SEGMENT_DETECT_RE = re.compile(
    r'(.*?)' +
    r'<loc(\d{4})>' * 4 + r'\s*' +
    '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
    r'\s*([^;<>]+)? ?(?:; )?'
)

def parse_instances(decoded: str):
    """
    Returns a list of dicts: {y1,x1,y2,x2 (int px), seg_indices (np.ndarray or None), label (str or None), span (str)}
    Supports multiple instances in one decoded string.
    """
    out = []
    text = decoded.lstrip("\n")
    while text:
        m = _SEGMENT_DETECT_RE.match(text)
        if not m:
            break
        gs = list(m.groups())
        before = gs.pop(0)  # any text before a match
        label = gs.pop()    # class name (may be None)
        y1, x1, y2, x2 = [int(x) / 1024.0 for x in gs[:4]]  # normalized
        seg_indices = gs[4:20]
        if seg_indices[0] is None:
            seg = None
        else:
            seg = np.array([int(x) for x in seg_indices], dtype=np.int32)
        content = m.group()
        if before:
            # carry through as non-object text if needed
            pass
        out.append(dict(y1=y1, x1=x1, y2=y2, x2=x2, seg_indices=seg, label=(label.strip() if label else None), span=content))
        text = text[len(content):]
    return out

# ---------- VQ-VAE mask decoder (JAX/Flax) ----------
# mirrors the reference implementation; it takes [B,16] codebook indices and returns [B,64,64,1] in [-1,1].
# Requires 'vae-oid.npz' alongside this script.
_VAE_PATH = "vae-oid.npz"

def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embeddings.shape[1]))
    return encodings

class _ResBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3,3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3,3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(1,1), padding=0)(x)
        return x + residual

class _Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        dim = 128
        x = nn.Conv(features=dim, kernel_size=(1,1), padding=0)(x)
        x = nn.relu(x)
        x = _ResBlock(features=dim)(x)
        x = _ResBlock(features=dim)(x)
        for _ in range(4):
            x = nn.ConvTranspose(features=dim, kernel_size=(4,4), strides=(2,2), padding=2, transpose_kernel=True)(x)
            x = nn.relu(x)
            dim //= 2
        x = nn.Conv(features=1, kernel_size=(1,1), padding=0)(x)
        return x

@jax.jit
def _decode_masks_jit(codebook_indices, params):
    quant = _quantized_values_from_codebook_indices(codebook_indices, params['_embeddings'])
    return _Decoder().apply({'params': params}, quant)

_params_cache = None
def _load_vae_params():
    global _params_cache
    if _params_cache is not None:
        return _params_cache
    arrs = dict(np.load(_VAE_PATH))
    def transp(kernel): return np.transpose(kernel, (2,3,1,0))
    def conv(name):
        return {'bias': arrs[name + '.bias'], 'kernel': transp(arrs[name + '.weight'])}
    def resblock(name):
        return {'Conv_0': conv(name + '.0'), 'Conv_1': conv(name + '.2'), 'Conv_2': conv(name + '.4')}
    _params_cache = {
        '_embeddings': arrs['_vq_vae._embedding'],
        'Conv_0': conv('decoder.0'),
        '_ResBlock_0': resblock('decoder.2.net'),
        '_ResBlock_1': resblock('decoder.3.net'),
        'ConvTranspose_0': conv('decoder.4'),
        'ConvTranspose_1': conv('decoder.6'),
        'ConvTranspose_2': conv('decoder.8'),
        'ConvTranspose_3': conv('decoder.10'),
        'Conv_1': conv('decoder.12'),
    }
    return _params_cache


def decode_seg_tokens_to_64x64(seg_indices: np.ndarray) -> np.ndarray:
    """
    seg_indices: shape (16,), int32 in [0,127]
    returns float32 mask in [0,1], shape (64,64)
    """
    params = _load_vae_params()
    m64 = _decode_masks_jit(seg_indices[None], params)[..., 0]  # [1,64,64]
    m64 = np.array(m64[0], dtype=np.float32)  # [-1,1]
    m64 = np.clip(m64 * 0.5 + 0.5, 0.0, 1.0)  # [0,1]
    return m64

def to_pixel_box(y1, x1, y2, x2, W, H):
    y1p = int(round(y1 * H)); x1p = int(round(x1 * W))
    y2p = int(round(y2 * H)); x2p = int(round(x2 * W))
    y1p = max(0, min(H, y1p)); y2p = max(0, min(H, y2p))
    x1p = max(0, min(W, x1p)); x2p = max(0, min(W, x2p))
    if y2p < y1p: y1p, y2p = y2p, y1p
    if x2p < x1p: x1p, x2p = x2p, x1p
    return y1p, x1p, y2p, x2p

def place_mask_fullres(H, W, box, m64, thresh=0.5):
    y1, x1, y2, x2 = box
    full = np.zeros((H, W), dtype=np.uint8)
    if (y2 > y1) and (x2 > x1):
        patch = Image.fromarray((m64 * 255.0).astype(np.uint8)).resize((x2 - x1, y2 - y1), resample=Image.BILINEAR)
        patch_arr = (np.array(patch, dtype=np.uint8) >= int(thresh * 255)).astype(np.uint8) * 255
        full[y1:y2, x1:x2] = patch_arr
    return full

def draw_bbox_overlay(pil_img: Image.Image, box, label=None, color=(255, 0, 0), width=3):
    im = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    y1, x1, y2, x2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            draw.text((x1 + 4, max(0, y1 - 18)), label, fill=color)
        except Exception:
            pass
    return im

def paligemma_postprocess(decoded: str, image: Image.Image):
    W, H = image.size
    instances = parse_instances(decoded)
    # Aggregate per label
    per_label_masks = {}
    overlay = image
    for inst in instances:
        box = to_pixel_box(inst['y1'], inst['x1'], inst['y2'], inst['x2'], W, H)
        overlay = draw_bbox_overlay(overlay, box, inst.get('label') or '')
        if inst['seg_indices'] is not None:
            m64 = decode_seg_tokens_to_64x64(inst['seg_indices'])
            mask = place_mask_fullres(H, W, box, m64, thresh=0.5)
            key = inst.get('label') or 'mask'
            if key not in per_label_masks:
                per_label_masks[key] = mask
            else:
                per_label_masks[key] = np.maximum(per_label_masks[key], mask)  # unio

    saved_masks = []
    cur_mask = None 
    for label, mask in per_label_masks.items():
            # mask_path = os.path.join(out_dir, f"{stem}__{safe_prompt if safe_prompt else 'segment'}_{label}.png")
            if cur_mask is not None:
                cur_mask = mask | cur_mask
            else:
                cur_mask = mask
            # saved_masks.append(mask_path)
    if cur_mask is not None:
        mask = Image.fromarray(cur_mask, mode='L')# .save(mask_path)
        return overlay, mask
    else:
        return None, None 
