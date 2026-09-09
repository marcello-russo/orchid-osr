"""Fast-repair Exp5 fase 2: allena la testa lineare su feature cached.

Equivalente matematico alla Fase 2 originale (backbone congelato, CE,
stessi iperparametri/seed) ma senza ripetere il forward del backbone a
ogni epoca: le feature vengono estratte una volta sola.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import torch  # noqa: E402

from app.cli.definitions import ALL_CLASSES, PROJECT_ROOT  # noqa: E402
from app.core.kfold.training import copy_fold_images, prepare_fold_data, split_raw_images  # noqa: E402
from app.core.models import create_model, get_registry_entry  # noqa: E402
from app.core.torch_metrics import MulticlassAccuracy, MulticlassF1Score  # noqa: E402
from app.preprocessing.reproducibility import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("repair_fast")

EXCLUDE = ["O. garganica", "O. majellensis"]
REPAIR = ["dinov2_base", "dinov2_small", "convnext_small", "resnet50"]
SEED = 57
BATCH = 8
CE_EPOCHS = 100


def extract_features(model, loader, device, normalize: bool):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            e = model.get_embeddings(inputs)
            if normalize:
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            feats.append(e.cpu())
            labs.append(labels)
    return torch.cat(feats), torch.cat(labs)


def head_module(model, variant: str):
    if variant.startswith("dinov2"):
        return model._head
    if variant.startswith("convnext"):
        return model._backbone.classifier[2]
    if variant.startswith("resnet"):
        return model._backbone.fc
    raise ValueError(variant)


def train_head(feats, labs, vfeats, vlabs, in_dim, device, lr, opt_type, wd):
    head = torch.nn.Linear(in_dim, 4).to(device)
    if opt_type == "adamw":
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    f1_m = MulticlassF1Score(num_classes=4, average="macro").to(device)
    acc_m = MulticlassAccuracy(num_classes=4, average="macro").to(device)
    best_f1, best_state, best_acc, best_ep = -1.0, None, 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    n = len(feats)
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(CE_EPOCHS):
        head.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            xb, yb = feats[idx].to(device), labs[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(head(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=2.0)
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        head.eval()
        with torch.no_grad():
            vout = head(vfeats.to(device))
            vloss = crit(vout, vlabs.to(device)).item()
            _, pred = torch.max(vout, 1)
            acc_m.reset() if hasattr(acc_m, "reset") else None
            f1_m.reset() if hasattr(f1_m, "reset") else None
            acc_m.update(pred.cpu(), vlabs)
            f1_m.update(pred.cpu(), vlabs)
            acc, f1 = acc_m.compute().item(), f1_m.compute().item()
        history["train_loss"].append(tot / cnt)
        history["val_loss"].append(vloss)
        history["val_accuracy"].append(acc)
        history["val_f1"].append(f1)
        if f1 >= best_f1:
            best_f1, best_acc, best_ep = f1, acc, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            log.info(f"  epoch {epoch+1}: loss={tot/cnt:.4f} val_f1={f1:.4f} (best {best_f1:.4f})")
    return best_state, best_f1, best_acc, best_ep, history


def main() -> None:
    only = sys.argv[1:] or REPAIR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    for variant in only:
        entry = get_registry_entry(variant)
        w, h = entry.input_size
        raw_dir = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / "train")
        tag = f"{w}x{h}_no_O._garganica_O._majellensis"
        exp_dir = BACKEND / "experiments" / f"exp5_sphor_ncd_{variant}"
        normalize = True
        lr = entry.learning_rate if entry.learning_rate else 0.001
        opt_type = entry.optimizer if entry.optimizer else "adam"
        wd = entry.weight_decay if entry.weight_decay else 0.0
        log.info(f"=== {variant} lr={lr} opt={opt_type} wd={wd} ===")
        all_paths, all_labels, folds = split_raw_images(raw_dir, list(ALL_CLASSES), 5, SEED, exclude_classes=EXCLUDE)
        train_names = [c for c in ALL_CLASSES if c not in EXCLUDE]
        for fold_idx in range(5):
            fold_json = exp_dir / f"fold_{fold_idx}.json"
            fj = json.loads(fold_json.read_text())
            hist = fj.get("history", {}) or {}
            if len(hist.get("val_f1", [])) > 5 and (fj.get("best_accuracy") or 0) > 0:
                log.info(f"fold {fold_idx}: fase 2 già presente, salto")
                continue
            shutil.copy(fold_json, str(fold_json) + ".pre_repair")
            train_idx, val_idx = folds[fold_idx]
            work = Path(f"/tmp/repair_exp5/{variant}/fold_{fold_idx}")
            if work.exists():
                shutil.rmtree(work)
            _, val_dir = copy_fold_images(all_paths, all_labels, train_names, train_idx, val_idx, str(work))
            aug_dir = str(PROJECT_ROOT / "dataset" / f"cropped_{w}x{h}" / f"kfold_augmented_{tag}" / f"fold_{fold_idx}")
            train_loader, val_loader = prepare_fold_data(aug_dir, val_dir, BATCH, SEED, variant, stratified=True)
            model = create_model(variant, num_classes=4, pretrained=True, normalize_embeddings=True)
            ckpt = str(exp_dir / f"best_fold_{fold_idx}.pt")
            model.load_weights(ckpt)
            model = model.to(device)
            feats, labs = extract_features(model, train_loader, device, normalize)
            vfeats, vlabs = extract_features(model, val_loader, device, normalize)
            log.info(f"fold {fold_idx}: feats {tuple(feats.shape)}")
            best_state, best_f1, best_acc, best_ep, history = train_head(
                feats, labs, vfeats, vlabs, feats.shape[1], device, lr, opt_type, wd
            )
            head_module(model, variant).load_state_dict(best_state)
            model.save_weights(ckpt)
            fj["best_f1"] = best_f1
            fj["best_epoch"] = best_ep
            fj["best_accuracy"] = best_acc
            fj["history"] = history
            fj["repaired_phase2_cached"] = True
            fold_json.write_text(json.dumps(fj, indent=2))
            log.info(f"fold {fold_idx}: RIPARATO best_f1={best_f1:.4f} acc={best_acc:.4f} ep={best_ep}")
    log.info("RIPARAZIONE COMPLETATA")


if __name__ == "__main__":
    main()
