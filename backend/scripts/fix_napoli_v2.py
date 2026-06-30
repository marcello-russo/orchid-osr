"""Re-letterbox Napoli images by detecting content bounding box."""

import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, ".")

from app.preprocessing.resizing import letterbox_resize


def content_bbox(img, threshold=30):
    """Find bounding box of non-background content in image."""
    arr = np.array(img)
    # Create mask: pixels where any channel > threshold
    mask = arr.sum(axis=2) > threshold * 3
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return (0, 0, img.width, img.height)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (int(xmin), int(ymin), int(xmax), int(ymax))


TARGETS = [
    ("dataset/cropped_256x512", 256, 512),
    ("dataset/cropped_224x224", 224, 224),
]

for base_dir, w, h in TARGETS:
    img_dir = os.path.join(base_dir, "test", "O. sphegodes_Napoli")
    if not os.path.isdir(img_dir):
        print(f"❌ {img_dir} non trovato")
        continue

    fixed = 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(img_dir, fname)

        with Image.open(fpath) as img:
            if img.size != (w, h):
                continue

            # Check if already has black bands
            top = img.crop((0, 0, img.width, 1))
            if any(sum(p) < 10 for p in list(top.getdata())[:w]):
                continue  # already letterboxed

            # Find content bounding box
            x1, y1, x2, y2 = content_bbox(img)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            # Crop to content + small padding
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            content = img.crop((x1, y1, x2, y2))

            # Re-letterbox from content crop
            result = letterbox_resize(content, w, h)

        result.save(fpath)
        fixed += 1

    print(f"✅ {base_dir}: {fixed} immagini riparate")
