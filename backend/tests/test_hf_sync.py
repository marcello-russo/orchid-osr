import pytest

from app.core.hf_sync import parse_exp_dir_name


@pytest.mark.parametrize(
    "dir_name,expected_exp,expected_model",
    [
        ("exp1_ce_resnet18", 1, "resnet18"),
        ("exp1_ce_resnet50", 1, "resnet50"),
        ("exp1_ce_convnext_tiny", 1, "convnext_tiny"),
        ("exp1_ce_convnext_small", 1, "convnext_small"),
        ("exp1_ce_dinov2_small", 1, "dinov2_small"),
        ("exp1_ce_dinov2_base", 1, "dinov2_base"),
        ("exp2_supcon_resnet18", 2, "resnet18"),
        ("exp2_supcon_resnet50", 2, "resnet50"),
        ("exp2_supcon_convnext_tiny", 2, "convnext_tiny"),
        ("exp2_supcon_convnext_small", 2, "convnext_small"),
        ("exp2_supcon_dinov2_small", 2, "dinov2_small"),
        ("exp2_supcon_dinov2_base", 2, "dinov2_base"),
        ("exp3_sphor_resnet18", 3, "resnet18"),
        ("exp3_sphor_resnet50", 3, "resnet50"),
        ("exp3_sphor_convnext_tiny", 3, "convnext_tiny"),
        ("exp3_sphor_convnext_small", 3, "convnext_small"),
        ("exp3_sphor_dinov2_small", 3, "dinov2_small"),
        ("exp3_sphor_dinov2_base", 3, "dinov2_base"),
        ("exp4_sphor_ncd_resnet18", 4, "resnet18"),
        ("exp4_sphor_ncd_resnet50", 4, "resnet50"),
        ("exp4_sphor_ncd_convnext_tiny", 4, "convnext_tiny"),
        ("exp4_sphor_ncd_convnext_small", 4, "convnext_small"),
        ("exp4_sphor_ncd_dinov2_small", 4, "dinov2_small"),
        ("exp4_sphor_ncd_dinov2_base", 4, "dinov2_base"),
        ("exp10_mgh_ncd_resnet18", 10, "resnet18"),
        ("exp10_mgh_ncd_resnet50", 10, "resnet50"),
        ("exp10_mgh_ncd_convnext_tiny", 10, "convnext_tiny"),
        ("exp10_mgh_ncd_convnext_small", 10, "convnext_small"),
        ("exp10_mgh_ncd_dinov2_small", 10, "dinov2_small"),
        ("exp10_mgh_ncd_dinov2_base", 10, "dinov2_base"),
    ],
)
def test_parse_exp_dir(dir_name, expected_exp, expected_model):
    """Verify exp directory name parsing matches expected exp_id and model."""
    exp_id, model = parse_exp_dir_name(dir_name)
    assert exp_id == expected_exp, f"{dir_name}: expected exp {expected_exp}, got {exp_id}"
    assert model == expected_model, f"{dir_name}: expected model {expected_model}, got {model}"
