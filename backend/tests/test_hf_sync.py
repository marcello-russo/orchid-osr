import pytest
from app.core.hf_sync import parse_exp_dir_name


@pytest.mark.parametrize("dir_name,expected_exp,expected_model", [
    ("exp6_clean_split_resnet18", 6, "resnet18"),
    ("exp6_clean_split_resnet50", 6, "resnet50"),
    ("exp6_clean_split_convnext_tiny", 6, "convnext_tiny"),
    ("exp6_clean_split_convnext_small", 6, "convnext_small"),
    ("exp6_clean_split_dinov2_small", 6, "dinov2_small"),
    ("exp6_clean_split_dinov2_base", 6, "dinov2_base"),
    ("exp7_supcon_resnet18", 7, "resnet18"),
    ("exp7_supcon_resnet50", 7, "resnet50"),
    ("exp7_supcon_convnext_tiny", 7, "convnext_tiny"),
    ("exp7_supcon_convnext_small", 7, "convnext_small"),
    ("exp7_supcon_dinov2_small", 7, "dinov2_small"),
    ("exp7_supcon_dinov2_base", 7, "dinov2_base"),
    ("exp8_sphor_resnet18", 8, "resnet18"),
    ("exp8_sphor_resnet50", 8, "resnet50"),
    ("exp8_sphor_convnext_tiny", 8, "convnext_tiny"),
    ("exp8_sphor_convnext_small", 8, "convnext_small"),
    ("exp8_sphor_dinov2_small", 8, "dinov2_small"),
    ("exp8_sphor_dinov2_base", 8, "dinov2_base"),
    ("exp9_ncd_resnet18", 9, "resnet18"),
    ("exp9_ncd_resnet50", 9, "resnet50"),
    ("exp9_ncd_convnext_tiny", 9, "convnext_tiny"),
    ("exp9_ncd_convnext_small", 9, "convnext_small"),
    ("exp9_ncd_dinov2_small", 9, "dinov2_small"),
    ("exp9_ncd_dinov2_base", 9, "dinov2_base"),
    ("exp10_mgh_ncd_resnet18", 10, "resnet18"),
    ("exp10_mgh_ncd_resnet50", 10, "resnet50"),
    ("exp10_mgh_ncd_convnext_tiny", 10, "convnext_tiny"),
    ("exp10_mgh_ncd_convnext_small", 10, "convnext_small"),
    ("exp10_mgh_ncd_dinov2_small", 10, "dinov2_small"),
    ("exp10_mgh_ncd_dinov2_base", 10, "dinov2_base"),
])
def test_parse_exp_dir(dir_name, expected_exp, expected_model):
    """Verify exp directory name parsing matches expected exp_id and model."""
    exp_id, model = parse_exp_dir_name(dir_name)
    assert exp_id == expected_exp, f"{dir_name}: expected exp {expected_exp}, got {exp_id}"
    assert model == expected_model, f"{dir_name}: expected model {expected_model}, got {model}"
