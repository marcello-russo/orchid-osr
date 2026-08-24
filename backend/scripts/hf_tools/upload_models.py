"""Upload only essential model files to HuggingFace (best fold + metadata).

Usage: python upload_models.py [--cleanup] [--exp all|exp6_ce|exp7_supcon]
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
from huggingface_hub import HfApi

from scripts.hf_config import HF_DATASET_REPO, HF_MODELS_REPO

# Carica HF_TOKEN dal .env (root progetto: resnet18/.env)
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    load_dotenv(_env)

HF_REPO = HF_MODELS_REPO
EXP_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
EXP_BASE = os.path.abspath(EXP_BASE)

MODELS = ["resnet18", "resnet50", "convnext_tiny", "convnext_small", "dinov2_small", "dinov2_base"]

VARIANT_DISPLAY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "dinov2_base": "DINOv2-Base",
}

EXPERIMENTS = [
    {"prefix": "exp6_clean_split", "exp": "exp6_ce", "label": "Exp 6 (CE)"},
    {"prefix": "exp7_supcon", "exp": "exp7_supcon", "label": "Exp 7 (SupCon)"},
    {"prefix": "exp8_sphor", "exp": "exp8_sphor", "label": "Exp 8 (SpHOR)"},
    {"prefix": "exp9_ncd", "exp": "exp9_ncd", "label": "Exp 9 (NCD)"},
    {"prefix": "exp10_mgh_ncd", "exp": "exp10_mgh_ncd", "label": "Exp 10 (MGH-NCD)"},
]


def upload_model(api, model, exp, exp_prefix):
    exp_name = f"{exp_prefix}_{model}"
    local_dir = os.path.join(EXP_BASE, exp_name)
    remote_dir = f"{exp}/{exp_name}"

    results_path = os.path.join(local_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"  SKIP {exp_name}: no results.json")
        return False

    results = json.load(open(results_path))
    best_fold = results.get("best_fold")
    if best_fold is None:
        print(f"  SKIP {exp_name}: no best_fold in results")
        return False

    best_pt = f"best_fold_{best_fold}.pt"
    if not os.path.exists(os.path.join(local_dir, best_pt)):
        print(f"  SKIP {exp_name}: {best_pt} not found")
        return False

    mean_f1 = results.get("mean_f1", 0)
    test_f1 = results.get("test_metrics", {}).get("f1_macro", 0)
    print(f"\n  Uploading {exp_name} (Val F1={mean_f1:.4f}, Test F1={test_f1:.4f})")

    files_to_upload = [
        ("results.json", "results.json"),
        ("config.json", "config.json"),
        (best_pt, best_pt),
    ]

    adapter_dir = os.path.join(local_dir, f"best_fold_{best_fold}_adapter")

    for local_name, remote_name in files_to_upload:
        local_path = os.path.join(local_dir, local_name)
        if not os.path.exists(local_path):
            continue
        size_kb = os.path.getsize(local_path) // 1024
        print(f"    {remote_name} ({size_kb}KB)")
        api.upload_file(
            repo_id=HF_REPO,
            path_or_fileobj=local_path,
            path_in_repo=f"{remote_dir}/{remote_name}",
        )

    if os.path.isdir(adapter_dir):
        print(f"    best_fold_{best_fold}_adapter/")
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=adapter_dir,
            path_in_repo=f"{remote_dir}/best_fold_{best_fold}_adapter",
        )

    emb_dir = os.path.join(local_dir, "embeddings")
    if os.path.isdir(emb_dir):
        print("    embeddings/")
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=emb_dir,
            path_in_repo=f"{remote_dir}/embeddings",
        )

    print(f"    Done: {exp_name}")
    return True


def cleanup_old_files(api):
    """Remove old fold weights and non-best fold files from HF."""
    print("\n  Cleaning up old files from HF...")
    all_files = api.list_repo_files(HF_REPO)
    removed = 0

    for exp_config in EXPERIMENTS:
        exp = exp_config["exp"]
        exp_prefix = exp_config["prefix"]

        for model in MODELS:
            exp_name = f"{exp_prefix}_{model}"
            prefix = f"{exp}/{exp_name}/"

            local_results = os.path.join(EXP_BASE, exp_name, "results.json")
            if not os.path.exists(local_results):
                continue
            best_fold = json.load(open(local_results)).get("best_fold")
            if best_fold is None:
                continue

            best_pt = f"best_fold_{best_fold}.pt"
            best_adapter = f"best_fold_{best_fold}_adapter"

            for f in all_files:
                if not f.startswith(prefix):
                    continue
                fname = f[len(prefix) :]

                if fname in ("results.json", "config.json", best_pt):
                    continue
                if fname.startswith(best_adapter):
                    continue

                if fname.startswith("best_fold_") and fname.endswith(".pt"):
                    print(f"    DELETE {f}")
                    api.delete_file(repo_id=HF_REPO, path_in_repo=f)
                    removed += 1
                elif fname.startswith("fold_") and fname.endswith(".json"):
                    print(f"    DELETE {f}")
                    api.delete_file(repo_id=HF_REPO, path_in_repo=f)
                    removed += 1
                elif fname.startswith("REPORT_"):
                    print(f"    DELETE {f}")
                    api.delete_file(repo_id=HF_REPO, path_in_repo=f)
                    removed += 1

    print(f"  Cleaned up {removed} old files")


def _build_results_table(exp, exp_prefix):
    rows = []
    for model in MODELS:
        exp_name = f"{exp_prefix}_{model}"
        local_results = os.path.join(EXP_BASE, exp_name, "results.json")
        if not os.path.exists(local_results):
            continue
        r = json.load(open(local_results))
        mean_f1 = r.get("mean_f1", 0)
        std_f1 = r.get("std_f1", 0)
        tm = r.get("test_metrics", {})
        test_acc = tm.get("accuracy", 0)
        test_f1 = tm.get("f1_macro", 0)
        best_fold = r.get("best_fold", "?")
        num_folds = r.get("num_folds", 5)
        display = VARIANT_DISPLAY.get(model, model)
        rows.append(
            f"| {display} | {mean_f1:.4f} ± {std_f1:.4f} | {test_f1:.4f} | {test_acc:.4f} | {best_fold} | {num_folds} |"
        )
    return rows


def _build_results_yaml(exp6_rows, exp7_rows, exp8_rows, exp9_rows=None, exp10_rows=None):
    """Generate structured YAML for model evaluation results."""
    lines = ["model-index:"]
    all_rows = []
    for exp_label, rows in [
        ("Exp 6 (CE)", exp6_rows),
        ("Exp 7 (SupCon)", exp7_rows),
        ("Exp 8 (SpHOR)", exp8_rows),
        ("Exp 9 (NCD)", exp9_rows or []),
        ("Exp 10 (Ultimate)", exp10_rows or []),
    ]:
        for row in rows:
            model_name = row.split("|")[1].strip()
            f1_val = row.split("|")[2].strip().split(" ")[0]
            all_rows.append((exp_label, model_name, f1_val))

    if not all_rows:
        return ""

    for i, (exp, model, f1) in enumerate(all_rows):
        lines.append(f"  - name: {model}")
        lines.append("    results:")
        lines.append("      - task:")
        lines.append("          type: image-classification")
        lines.append("        dataset:")
        lines.append(f"          name: {HF_DATASET_REPO.rsplit('/', 1)[-1]}")
        lines.append(f"          type: {HF_DATASET_REPO}")
        lines.append("        metrics:")
        lines.append("          - name: Macro F1 Score")
        lines.append("            type: f1_macro")
        lines.append(f"            value: {f1}")
        lines.append("        source:")
        lines.append(f"          name: {exp}")
        lines.append(f"          url: https://huggingface.co/datasets/{HF_DATASET_REPO}")

    return "\n".join(lines)


def update_readme(api):
    """Update HF model repo README with current results."""
    header = "| Model | Val F1 (macro) | Test F1 | Test Acc | Best Fold | Folds |\n|---|---|---|---|---|---|"

    exp6_rows = _build_results_table("exp6_ce", "exp6_clean_split")
    exp7_rows = _build_results_table("exp7_supcon", "exp7_supcon")
    exp8_rows = _build_results_table("exp8_sphor", "exp8_sphor")
    exp9_rows = _build_results_table("exp9_ncd", "exp9_ncd")
    exp10_rows = _build_results_table("exp10_mgh_ncd", "exp10_mgh_ncd")

    sections = []

    if exp6_rows:
        sections.append(f"""## Exp 6: Cross-Entropy

5-fold stratified cross-validation on deduplicated clean split (2,232 train, 300 test).

**Training recipe:** epochs=100, patience=15, effective batch=32
(gradient accumulation ×4), per-architecture LR from registry.

{header}
{chr(10).join(exp6_rows)}""")

    if exp7_rows:
        sections.append(f"""## Exp 7: Supervised Contrastive Learning

Two-phase training: SupCon pretraining (InfoNCE, τ=0.07) → CE fine-tuning with frozen backbone.

**Recipe:** same per-architecture optimizer/LR, projection dim=128, CE Phase LR=0.01, patience=15 on val metrics.

{header}
{chr(10).join(exp7_rows)}""")

    if exp8_rows:
        sections.append(f"""## Exp 8: Spherical Orthogonal Prototypes (SpHOR)

Two-phase training: SupCon + Spherical Orthogonal Prototypes → CE fine-tuning with frozen backbone.

**Recipe:** same as Exp 7, with spherical prototype repulsion (repulse=0.01).

{header}
{chr(10).join(exp8_rows)}""")

    if exp9_rows:
        sections.append(f"""## Exp 9: Novel Class Discovery (NCD)

NCD scenario: O. majellensis excluded from training (hidden novel class).

**Recipe:** same as Exp 8.

{header}
{chr(10).join(exp9_rows)}""")

    if exp10_rows:
        sections.append(f"""## Exp 10: Ultimate Experiment

Final experiment combining multi-granularity features, hard negative mining, deep fine-tuning, and OSR ensemble.

**Recipe:** LoRA r=32 + partial unfreeze for DINOv2, full fine-tuning for
ConvNeXt/ResNet. Hard negative weight 5.0 on (majellensis, sphegodes) pairs.

{header}
{chr(10).join(exp10_rows)}""")

    results_yaml = _build_results_yaml(exp6_rows, exp7_rows, exp8_rows, exp9_rows, exp10_rows)

    readme = f"""---
library_name: pytorch
tags:
- computer-vision
- image-classification
- fine-grained-classification
- ophrys-orchids
- resnet
- convnext
- dinov2
datasets:
- {HF_DATASET_REPO}
license: mit
pipeline_tag: image-classification
{results_yaml}
---

# OrchID OSR-NCD Models

Trained model weights for the [OrchID OSR-NCD](https://huggingface.co/spaces/marcellorusso/orchid-osr)
project — ultra-fine-grained visual classification of *Ophrys* orchids.

Fine-grained classification of six cryptic *Ophrys* species
(*O. exaltata, O. garganica, O. incubacea, O. majellensis, O. sphegodes, O. sphegodes* Palena)
using ResNet-18/50, ConvNeXt-Tiny/Small, and DINOv2-Small/Base.

{chr(10).join(sections)}

## Structure

```
exp6_ce/                           — Exp 6 (Cross-Entropy)
  exp6_clean_split_resnet18/
    results.json                   — aggregated metrics + test results
    config.json                    — training configuration
    best_fold_N.pt                 — weights of the best fold
    best_fold_N_adapter/           — LoRA adapter (DINOv2 only)
exp7_supcon/                       — Exp 7 (SupCon + CE fine-tune)
  exp7_supcon_resnet18/
    ...
exp8_sphor/                        — Exp 8 (SpHOR)
  exp8_sphor_resnet18/
    ...
exp9_ncd/                          — Exp 9 (NCD)
  exp9_ncd_resnet18/
    ...
exp10_mgh_ncd/                     — Exp 10 (MGH-NCD)
  exp10_mgh_ncd_resnet18/
    ...
```

## Usage

The classifier in the [OrchID OSR-NCD Space](https://huggingface.co/spaces/marcellorusso/orchid-osr)
downloads these weights at startup and uses them for inference.

## Links

- [Dataset](https://huggingface.co/datasets/{HF_DATASET_REPO})
- [Live Demo](https://huggingface.co/spaces/marcellorusso/orchid-osr)
- [GitHub](https://github.com/squidslab/OrchID OSR)
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        f.flush()
        api.upload_file(
            repo_id=HF_REPO,
            path_or_fileobj=f.name,
            path_in_repo="README.md",
        )
        os.unlink(f.name)
    print("  README.md updated")


def main():
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.whoami()
    print("Authenticated OK")

    do_cleanup = "--cleanup" in sys.argv

    # Determine which experiments to upload
    exp_filter = "all"
    for arg in sys.argv:
        if arg.startswith("--exp="):
            exp_filter = arg.split("=")[1]
        elif arg == "--exp" and sys.argv.index(arg) + 1 < len(sys.argv):
            exp_filter = sys.argv[sys.argv.index(arg) + 1]

    uploaded = 0
    total = 0
    for exp_config in EXPERIMENTS:
        exp = exp_config["exp"]
        exp_prefix = exp_config["prefix"]
        label = exp_config["label"]

        if exp_filter != "all" and exp_filter not in exp:
            continue

        print(f"\n=== Uploading {label} ({exp}) ===")
        for model in MODELS:
            total += 1
            try:
                if upload_model(api, model, exp, exp_prefix):
                    uploaded += 1
            except Exception as e:
                print(f"  ERROR {model}: {e}")

    print(f"\nUploaded {uploaded}/{total} models")

    if do_cleanup:
        cleanup_old_files(api)

    update_readme(api)

    print("\nDone!")


if __name__ == "__main__":
    main()
