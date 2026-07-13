"""Tests for _find_weights in weights.py."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_experiment_dir(base, name, model_variant, best_fold=0, f1=0.9):
    """Create a fake experiment directory with results.json and a dummy .pt file."""
    exp_dir = Path(base) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_variant": model_variant,
        "best_fold": best_fold,
        "mean_f1": f1,
        "num_folds": 5,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f)
    
    # Create dummy .pt file for the best fold
    pt_file = exp_dir / f"best_fold_{best_fold}.pt"
    pt_file.write_text("dummy weights")
    
    return exp_dir


def test_find_weights_selects_best_f1():
    """Should select the experiment with highest F1 for a model type."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_experiment_dir(tmp, "exp6_clean_split_resnet18", "resnet18", f1=0.88)
        _make_experiment_dir(tmp, "exp7_supcon_resnet18", "resnet18", f1=0.92)
        _make_experiment_dir(tmp, "exp8_sphor_resnet18", "resnet18", f1=0.75)
        
        with patch("app.api.weights.settings") as mock_settings:
            mock_settings.results_dir = tmp
            mock_settings.hf_token = ""
            
            from app.api.weights import _find_weights
            result = _find_weights("resnet18")
            
            assert result is not None
            assert "exp7_supcon_resnet18" in str(result)


def test_find_weights_skips_exp9():
    """Exp 9 experiments should be excluded from results."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_experiment_dir(tmp, "exp9_ncd_resnet18", "resnet18", f1=0.95)
        _make_experiment_dir(tmp, "exp8_sphor_resnet18", "resnet18", f1=0.90)
        
        with patch("app.api.weights.settings") as mock_settings:
            mock_settings.results_dir = tmp
            mock_settings.hf_token = ""
            
            from app.api.weights import _find_weights
            result = _find_weights("resnet18")
            
            assert result is not None
            assert "exp9_ncd" not in str(result)
            assert "exp8_sphor" in str(result)


def test_find_weights_returns_none_when_no_match():
    """Should return None when no experiments match the model type."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_experiment_dir(tmp, "exp6_clean_split_resnet50", "resnet50")
        
        with patch("app.api.weights.settings") as mock_settings:
            mock_settings.results_dir = tmp
            mock_settings.hf_token = ""
            
            from app.api.weights import _find_weights
            result = _find_weights("resnet18")
            
            assert result is None


def test_find_weights_handles_missing_pt():
    """Should skip experiments where the .pt file is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = _make_experiment_dir(tmp, "exp6_clean_split_resnet18", "resnet18", f1=0.9)
        # Delete the .pt file
        (exp_dir / "best_fold_0.pt").unlink()
        
        with patch("app.api.weights.settings") as mock_settings:
            mock_settings.results_dir = tmp
            mock_settings.hf_token = ""
            
            from app.api.weights import _find_weights
            result = _find_weights("resnet18")
            
            assert result is None
