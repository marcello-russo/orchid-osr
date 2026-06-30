"""Re-crop stretched Napoli images by recovering flower bounding box."""

import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, ".")
from app.preprocessing.resizing import letterbox_resize


def find_flower_bbox(img):
    """Find the flower's bounding box using color/saliency."""
    arr = np.array(img)
    # Convert to HSV-like: find green/colorful pixels
    # Flowers are typically the most saturated/vibrant region
    h, w = arr.shape[:2]
    
    # Simple saliency: color variance per pixel
    std_map = arr.std(axis=2)
    mask = std_map > 25  # colorful pixels (not gray/brown background)
    
    # Also exclude very dark or very bright edges
    brightness = arr.mean(axis=2)
    mask &= (brightness > 20) & (brightness < 240)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return (0, 0, w, h)
    
    # Add some margin
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y1 = max(0, y_idx[0] - 5)
    y2 = min(h, y_idx[-1] + 5)
    x1 = max(0, x_idx[0] - 5)
    x2 = min(w, x_idx[-1] + 5)
    
    return (x1, y1, x2, y2)


TARGETS = [
    ("dataset/cropped_256x512", 256, 512),
    ("dataset/cropped_224x224", 224, 224),
]

for base_dir, tw, th in TARGETS:
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
            if img.size != (tw, th):
                continue

            # Skip if already letterboxed
            top = img.crop((0, 0, img.width, 1))
            if any(sum(p) < 10 for p in list(top.getdata())[:img.width]):
                continue

            # Find flower bounding box
            x1, y1, x2, y2 = find_flower_bbox(img)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            # Extract flower
            flower = img.crop((x1, y1, x2, y2))
            
            # Letterbox to target size
            result = letterbox_resize(flower, tw, th)

        result.save(fpath)
        fixed += 1

    print(f"✅ {base_dir}: {fixed} immagini riparate")
