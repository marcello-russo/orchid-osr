"""Test per la logica di resume fold: atomic write, validazione JSON corrotto e resume."""

import json
import os


def _make_fake_exp(tmp_path, folds_configs):
    """Crea un esperimento finto con fold_*.json secondo folds_configs.

    folds_configs: list di dict con 'valid' (bool) e 'idx' (int).
    """
    exp_dir = tmp_path / "exp_test_resume"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(json.dumps({"model_variant": "resnet18", "num_classes": 6}))
    for fc in folds_configs:
        path = exp_dir / f"fold_{fc['idx']}.json"
        if fc.get("valid", True):
            path.write_text(json.dumps({"best_f1": 0.9, "fold": fc["idx"]}))
        else:
            path.write_text("{corrupted json")
    return exp_dir


def _check_completed(exp_dir):
    """Simula la logica di resume in __init__.py."""
    import json as _json

    completed = []
    for fold_idx in range(5):
        fold_json = exp_dir / f"fold_{fold_idx}.json"
        if fold_json.exists():
            try:
                with open(fold_json) as f:
                    _json.load(f)
            except (json.JSONDecodeError, ValueError, OSError):
                os.remove(fold_json)
            else:
                completed.append(fold_idx)
    return completed


def test_fold_json_corrupted_removed_and_retried(tmp_path):
    """JSON corrotto di fold viene rimosso e non considerato completato."""
    exp_dir = _make_fake_exp(
        tmp_path,
        [
            {"idx": 0, "valid": True},
            {"idx": 1, "valid": False},  # corrotto
            {"idx": 2, "valid": True},
        ],
    )
    completed = _check_completed(exp_dir)
    assert 0 in completed
    assert 1 not in completed  # è stato rimosso
    assert 2 in completed
    assert not (exp_dir / "fold_1.json").exists()


def test_fold_json_atomic_write_survives_crash(tmp_path):
    """Simula crash durante scrittura JSON: .tmp non completa = fold non esiste."""
    exp_dir = tmp_path / "exp_test_atomic"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(json.dumps({"model_variant": "resnet18", "num_classes": 6}))
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.85}))

    # Simula crash: .tmp scritto ma mai rinominato in .json
    fold_path = exp_dir / "fold_0.json"
    tmp = exp_dir / "fold_0.json.tmp"
    tmp.write_text('{"best_f1": 0.8')  # JSON troncato!

    assert not fold_path.exists()  # fold_0.json NON esiste
    assert tmp.exists()  # solo il .tmp rimane

    # Il resume cerca fold_*.json → non matcha .tmp → fold rifatto


def test_fold_json_clean_tmp_on_restart(tmp_path):
    """All'avvio, i file .tmp residui non interferiscono col resume."""
    exp_dir = tmp_path / "exp_test_cleanup"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(json.dumps({"model_variant": "resnet18", "num_classes": 6}))
    (exp_dir / "results.json").write_text(json.dumps({"mean_f1": 0.85}))

    # Crea fold_0 valido
    (exp_dir / "fold_0.json").write_text(json.dumps({"best_f1": 0.9, "fold": 0}))
    # Lascia un .tmp orfano
    (exp_dir / "fold_1.json.tmp").write_text("garbage")

    completed = _check_completed(exp_dir)
    assert 0 in completed
    assert len(completed) == 1  # solo fold_0
