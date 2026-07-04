"""Test per early stopping in SupCon training.

Verifica che il contatore epochs_no_improve si incrementi
correttamente quando la metrica plateaus (non solo quando cala).
"""

import pytest


def _simulate_supcon_epochs(knn_history, patience=5, use_strict_gt=True):
    """Simula la logica di early stopping di supcon.py.

    Prima del fix (>=): increments solo su calo.
    Dopo il fix (>): incrementa anche su plateau.
    """
    best_knn = -1.0
    epochs_no_improve = 0
    best_epoch = None
    early_stopped = False

    for epoch, knn_acc in enumerate(knn_history):
        if use_strict_gt:
            if knn_acc > best_knn:
                best_knn = knn_acc
                epochs_no_improve = 0
                best_epoch = epoch
            else:
                epochs_no_improve += 1
        else:
            if knn_acc >= best_knn:
                best_knn = knn_acc
                epochs_no_improve = 0
                best_epoch = epoch
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            early_stopped = True
            break

    return epochs_no_improve, early_stopped, best_epoch


def test_old_bug_plateau_never_increments_counter():
    """OLD: >= resetta contatore anche su plateau → non si ferma mai su plateau."""
    # 4 epoche di plateau dopo epoch 1, ma il contatore non cresce
    k = [0.8, 0.85, 0.85, 0.85, 0.85]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=False)
    assert eps == 0, f"Bug: plateau dovrebbe resettare contatore, ma eps={eps}"
    assert not stopped

    # Finché la metrica non cala, non si ferma
    k2 = [0.8, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82]
    eps2, stopped2, _ = _simulate_supcon_epochs(k2, patience=5, use_strict_gt=False)
    assert eps2 == 0
    assert not stopped2


def test_fix_plateau_increments_counter():
    """FIX: > non resetta su plateau → contatore cresce → early stopping funziona."""
    k = [0.8, 0.85, 0.85, 0.85, 0.85]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=True)
    # epoche: 0=0.8(migliora) 1=0.85(migliora) 2=0.85(plateau→+1) 3=0.85(→+2) 4=0.85(→+3≥3→stop)
    assert eps >= 3
    assert stopped


def test_fix_strict_improvement_needed():
    """FIX: stesso valore non è miglioramento, contatore sale."""
    k = [0.8, 0.85, 0.85, 0.86, 0.86, 0.86, 0.86, 0.86]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=True)
    # 0.86 a epoch 3, poi 4 epoche a 0.86 (patience=3) → stop
    assert stopped
    assert eps >= 3


def test_fix_improvement_resets_counter():
    """FIX: miglioramento stretto resetta contatore."""
    k = [0.8, 0.82, 0.85, 0.84, 0.87]
    eps, stopped, best = _simulate_supcon_epochs(k, patience=2, use_strict_gt=True)
    assert not stopped
    assert best == 4


def test_fix_oscillation_stops_eventually():
    k = [0.8, 0.85, 0.83, 0.84, 0.82]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=2, use_strict_gt=True)
    # dopo 0.85, abbiamo: 0.83(→+1) 0.84(→+2≥2→stop)
    assert stopped


def test_fix_constant_values_stop():
    """Sempre lo stesso valore → early stopping dopo patience epoche."""
    k = [0.7, 0.7, 0.7, 0.7]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=True)
    assert stopped


def test_fix_decreasing_stops():
    """Valori in calo costante → early stopping."""
    k = [0.8, 0.75, 0.72, 0.73, 0.71]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=True)
    # 0.8 epoch0, 0.75→+1, 0.72→+2, 0.73(non>0.8)→+3≥3→stop
    assert stopped


def test_fix_short_history_no_stop():
    """Storia troppo breve → no early stopping."""
    k = [0.7]
    eps, stopped, _ = _simulate_supcon_epochs(k, patience=3, use_strict_gt=True)
    assert not stopped


class TestOldBehaviorVsFix:
    """Confronta diretto: vecchio >= vs nuovo >."""

    @pytest.mark.parametrize(
        "history,patience,expect_old_stop,expect_new_stop",
        [
            ([0.8, 0.85, 0.85, 0.85, 0.85], 3, False, True),  # plateau: old non ferma, fix sì
            ([0.7, 0.8, 0.9], 3, False, False),  # sempre migliora: nessuno ferma
            ([0.8, 0.85, 0.86, 0.86, 0.86, 0.86], 3, False, True),  # plateau dopo miglioramento
            ([0.8, 0.75, 0.72, 0.7], 2, True, True),  # calo: entrambi fermano
        ],
    )
    def test_comparison(self, history, patience, expect_old_stop, expect_new_stop):
        _, old_stopped, _ = _simulate_supcon_epochs(history, patience, use_strict_gt=False)
        _, new_stopped, _ = _simulate_supcon_epochs(history, patience, use_strict_gt=True)
        assert old_stopped == expect_old_stop
        assert new_stopped == expect_new_stop
