"""Sanity check: verify NCD pipeline with known vs novel class."""

import json
import sys
import torch
import numpy as np

from app.core.ncd.extraction import extract_all
from app.core.ncd.detector import get_detector, list_detectors
from app.core.ncd.evaluation import evaluate_novel
from app.core.models import get_registry_entry

device = torch.device("cuda")
exp_dir = "experiments/exp10_mgh_ncd_dinov2_base"
reg = get_registry_entry("dinov2_base")
w, h = reg.input_size
data_dir = f"dataset/cropped_{w}x{h}/train"
test_dir = f"dataset/cropped_{w}x{h}/test"

results_path = f"{exp_dir}/results.json"
best_fold = json.load(open(results_path)).get("best_fold", 0)
model_path = f"{exp_dir}/best_fold_{best_fold}.pt"

collect = any(get_detector(n).requires_logits for n in list_detectors())
print("Extracting embeddings...", flush=True)
bundle = extract_all(
    "dinov2_base", data_dir, test_dir, model_path, 6, 8, device,
    collect_logits=collect, exp_dir=exp_dir,
)
print("Done.", flush=True)

print("\n=== Test 1: Napoli come novel (MAI vista in training) ===", flush=True)
for det_name in list_detectors():
    det = get_detector(det_name)
    det.fit(bundle.train_embeddings, bundle.train_labels)
    scores = det.score(
        bundle.test_embeddings,
        logits=bundle.test_logits if det.requires_logits else None,
    )
    result = evaluate_novel(
        scores, bundle.test_labels, "O. sphegodes_Napoli", bundle.test_class_names,
    )
    print(f"  {det_name:15s}: AUROC = {result['auroc']:.4f}", flush=True)

print("\n=== Test 2: Exaltata come novel (VISTA in training) ===", flush=True)
for det_name in list_detectors():
    det = get_detector(det_name)
    det.fit(bundle.train_embeddings, bundle.train_labels)
    scores = det.score(
        bundle.test_embeddings,
        logits=bundle.test_logits if det.requires_logits else None,
    )
    result = evaluate_novel(
        scores, bundle.test_labels, "O. exaltata", bundle.test_class_names,
    )
    print(f"  {det_name:15s}: AUROC = {result['auroc']:.4f}", flush=True)

print("\n✅ Sanity check completato", flush=True)
