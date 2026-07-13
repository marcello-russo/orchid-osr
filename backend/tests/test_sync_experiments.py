"""Test for sync_experiments._is_essential — filters files for HF sync."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sync_experiments import _is_essential


def test_essential_json():
    """results.json must be synced."""
    assert _is_essential("exp8_sphor_resnet18/results.json") is True


def test_essential_config_json():
    """config.json must be synced."""
    assert _is_essential("exp8_sphor_resnet18/config.json") is True


def test_essential_ncd_results():
    """ncd_results.json must be synced."""
    assert _is_essential("exp8_sphor_resnet18/ncd_results.json") is True


def test_essential_osr_results():
    """osr_results.json must be synced."""
    assert _is_essential("exp8_sphor_resnet18/osr_results.json") is True


def test_essential_embeddings_npy():
    """Embedding .npy files must be synced (needed for latent space viewer)."""
    assert _is_essential("exp8_sphor_resnet18/embeddings/tsne_2d.npy") is True


def test_essential_embeddings_labels():
    """Embedding labels.npy must be synced."""
    assert _is_essential("exp8_sphor_resnet18/embeddings/labels.npy") is True


def test_essential_embeddings_metadata():
    """Embedding metadata.json must be synced."""
    assert _is_essential("exp8_sphor_resnet18/embeddings/metadata.json") is True


def test_skip_pt_weights():
    """.pt weights must NOT be synced (handled on-demand by weights.py)."""
    assert _is_essential("exp8_sphor_resnet18/best_fold_0.pt") is False


def test_skip_safetensors():
    """.safetensors files (LoRA adapters) must NOT be synced."""
    assert _is_essential("exp8_sphor_resnet18/best_fold_0_adapter/adapter_model.safetensors") is False


def test_skip_ncd_figures():
    """PNG figures from NCD analysis must NOT be synced."""
    assert _is_essential("exp8_sphor_resnet18/ncd_figures/tsne.png") is False


def test_skip_old_prefix():
    """Files under 'old/' prefix must NOT be synced."""
    assert _is_essential("old/exp6_ce/results.json") is False


def test_skip_cache_dir():
    """Files under '.cache/' must NOT be synced."""
    assert _is_essential("exp8_sphor_resnet18/.cache/something.json") is False


def test_ncd_comparison_json():
    """Root-level ncd_comparison.json must be synced."""
    assert _is_essential("ncd_comparison.json") is True
