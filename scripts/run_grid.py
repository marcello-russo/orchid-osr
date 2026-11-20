#!/usr/bin/env python3
"""Grid completo con la variante scelta dall'ablazione.

Legge `ablation_decision.json` (scritto da `ablation_sphor.py --decide`) e
addestra gli esperimenti richiesti con quei parametri, un modello alla volta.
Se il verdetto dice `run_grid: false` non fa niente: il metodo non ha battuto
SupCon e non c'e' niente da promuovere.

Uso:
    scripts/run_grid.py --experiments 4,5 --models dinov2_small,dinov2_base,convnext_tiny
    scripts/run_grid.py --experiments 3 --models resnet18,resnet50,convnext_small

Le fold gia' completate vengono saltate (`kfold_experiment` guarda i fold JSON),
quindi il comando si puo' rilanciare dopo una pausa o un crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_sphor import VARIANTS  # noqa: E402

DEFAULT_MODELS = "dinov2_small,dinov2_base,convnext_tiny"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="4,5")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--variant", default=None, help="Default: quella scelta dall'ablazione")
    parser.add_argument("--tag", default="sphor-v2")
    # Protocollo deciso dall'ablazione: budget fisso e modello finale. Con
    # l'early stopping la KNN sceglie l'epoca del warm-up (misurato: 0.9172
    # contro 0.9049 della fase congiunta) e si spedirebbe SupCon chiamandolo
    # SpHOR; l'AUROC invece premia la fine della fase congiunta.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ce-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--restore-best", action="store_true", help="Torna alla selezione per KNN (default: spento)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    decision_path = root / "ablation_decision.json"
    if not decision_path.exists():
        print("manca ablation_decision.json: lancia prima `ablation_sphor.py --decide`", flush=True)
        return 2
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("run_grid"):
        print(f"il verdetto dice di NON lanciare il grid: {decision.get('reason')}", flush=True)
        return 0

    variant = args.variant or decision["winner"]
    params = dict(VARIANTS[variant])
    experiments = [int(x) for x in args.experiments.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"=== variante {variant} {params} ===", flush=True)
    print(f"=== esperimenti {experiments} · modelli {models} · tag {args.tag} ===", flush=True)

    from app.cli.runner import run_experiment

    failures: list[str] = []
    started = time.time()
    for exp in experiments:
        for model in models:
            label = f"exp{exp}/{model}"
            t0 = time.time()
            print(f"=== {label} inizio {time.strftime('%Y-%m-%d %H:%M')} ===", flush=True)
            try:
                run_experiment(
                    exp_id=exp,
                    model_variant=model,
                    epochs=args.epochs,
                    ce_epochs=args.ce_epochs,
                    patience=args.patience,
                    restore_best=args.restore_best,
                    tag=args.tag,
                    **params,
                )
                print(f"=== {label} finito in {(time.time() - t0) / 3600:.2f} h ===", flush=True)
            except Exception:
                failures.append(label)
                print(f"=== {label} FALLITO ===", flush=True)
                traceback.print_exc()

    print(f"=== grid finito in {(time.time() - started) / 3600:.2f} h, falliti: {failures} ===", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
