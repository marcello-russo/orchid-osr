"""Re-crop Napoli images from raw originals with proper letterboxing."""

import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, ".")
from app.preprocessing.resizing import letterbox_resize


def find_flower_bbox_raw(img):
    """Find flower center using simple center-weighted crop."""
    # Raw photos typically have the flower centered. Just center-crop at 3:4 ratio.
    iw, ih = img.size
    # Target aspect ratio: 3:4 (typical orchid flower)
    target_ar = 3 / 4
    if iw / ih > target_ar:
        # Too wide: crop width
        new_w = int(ih * target_ar)
        x = (iw - new_w) // 2
        return (x, 0, x + new_w, ih)
    else:
        # Too tall: crop height
        new_h = int(iw / target_ar)
        y = (ih - new_h) // 2
        return (0, y, iw, y + new_h)


RAW_DIR = "dataset/raw/O. sphegodes_Napoli"
TARGETS = [
    ("dataset/cropped_256x512/test/O. sphegodes_Napoli", 256, 512),
    ("dataset/cropped_224x224/test/O. sphegodes_Napoli", 224, 224),
]

import glob

# Build mapping from raw filename stem → raw path
raw_files = {}
for f in os.listdir(RAW_DIR):
    stem = os.path.splitext(f)[0]
    raw_files[stem] = os.path.join(RAW_DIR, f)

for target_dir, tw, th in TARGETS:
    fixed = 0
    errors = 0
    for fname in sorted(os.listdir(target_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(target_dir, fname)
        stem = os.path.splitext(fname)[0]

        # Find matching raw image
        raw_path = raw_files.get(stem)
        if raw_path is None:
            errors += 1
            continue

        with Image.open(fpath) as cropped_img:
            if cropped_img.size != (tw, th):
                continue
            # Skip if already letterboxed
            top = cropped_img.crop((0, 0, cropped_img.width, 1))
            if any(sum(p) < 10 for p in list(top.getdata())[:cropped_img.width]):
                continue

        try:
            with Image.open(raw_path) as raw_img:
                # Find flower in raw image
                bbox = find_flower_bbox_raw(raw_img)
                flower = raw_img.crop(bbox)
                # Letterbox to target
                result = letterbox_resize(flower, tw, th)
            result.save(fpath)
            fixed += 1
        except Exception as e:
            errors += 1

    print(f"✅ {target_dir}: {fixed} riparate, {errors} errori")
