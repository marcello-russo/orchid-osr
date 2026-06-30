"""Re-letterbox Napoli images that were saved stretched."""

import os
import sys
from PIL import Image

sys.path.insert(0, ".")

from app.preprocessing.resizing import letterbox_resize

TARGETS = [
    ("dataset/cropped_256x512", 256, 512),
    ("dataset/cropped_224x224", 224, 224),
]

napoli_dir = "O. sphegodes_Napoli"

for base_dir, w, h in TARGETS:
    img_dir = os.path.join(base_dir, "test", napoli_dir)
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
                print(f"  ⚠ {fname}: size {img.size}, skip")
                continue
            # Check if already letterboxed (top row all black)
            top = img.crop((0, 0, img.width, 1))
            top_pixels = list(top.getdata())
            has_bands = any(sum(p) < 10 for p in top_pixels[:w])
            if has_bands:
                continue  # già letterboxed

            # Re-apply letterbox resize
            fixed_img = letterbox_resize(img, w, h)

        fixed_img.save(fpath)
        fixed += 1

    print(f"✅ {base_dir}/{napoli_dir}: {fixed} immagini ri-letterboxate")
