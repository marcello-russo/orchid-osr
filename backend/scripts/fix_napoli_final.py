"""Re-crop all stretched Napoli images from raw originals."""

import os
import sys
from PIL import Image
sys.path.insert(0, ".")
from app.preprocessing.resizing import letterbox_resize

RAW_DIR = "dataset/raw/O. sphegodes_Napoli"
TARGETS = [
    ("dataset/cropped_256x512/test/O. sphegodes_Napoli", 256, 512),
    ("dataset/cropped_224x224/test/O. sphegodes_Napoli", 224, 224),
]

# Build filename map
ext_map = {}
for f in os.listdir(RAW_DIR):
    stem = os.path.splitext(f)[0]
    ext_map[stem] = os.path.join(RAW_DIR, f)

for target_dir, tw, th in TARGETS:
    all_files = sorted(os.listdir(target_dir))
    fixed, skipped, errors = 0, 0, 0
    
    for fname in all_files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(target_dir, fname)
        stem = os.path.splitext(fname)[0]

        with Image.open(fpath) as img:
            if img.size != (tw, th):
                continue
            # Skip if already letterboxed
            top_px = list(img.crop((0, 0, tw, 1)).getdata())
            if any(sum(p) < 10 for p in top_px[:tw]):
                skipped += 1
                continue

        raw_path = ext_map.get(stem)
        if raw_path is None:
            errors += 1
            continue

        try:
            with Image.open(raw_path) as raw_img:
                iw, ih = raw_img.size
                # Center crop at 3:4 aspect ratio
                if iw / ih > 3 / 4:
                    new_w = int(ih * 3 / 4)
                    x = (iw - new_w) // 2
                    flower = raw_img.crop((x, 0, x + new_w, ih))
                else:
                    new_h = int(iw * 4 / 3)
                    y = (ih - new_h) // 2
                    flower = raw_img.crop((0, y, iw, y + new_h))
                result = letterbox_resize(flower, tw, th)
            result.save(fpath)
            fixed += 1
        except Exception as e:
            errors += 1

        if (fixed + skipped + errors) % 100 == 0:
            print(f"  {target_dir}: {fixed} fixed, {skipped} skipped, {errors} errors")

    print(f"✅ {target_dir}: {fixed} fixed, {skipped} skipped, {errors} errors")
