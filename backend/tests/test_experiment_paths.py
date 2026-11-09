from app.core.experiment_paths import hf_exp_dir, hf_file


def test_exp2_uses_remote_exp7_name():
    assert hf_exp_dir("exp2_supcon_convnext_small") == "exp7_supcon/exp7_supcon_convnext_small"
    assert hf_file("exp2_supcon_convnext_small", "best_fold_1.pt") == (
        "exp7_supcon/exp7_supcon_convnext_small/best_fold_1.pt"
    )


def test_all_mappings():
    assert hf_exp_dir("exp1_ce_resnet18") == "exp6_ce/exp6_ce_resnet18"
    assert hf_exp_dir("exp3_sphor_dinov2_base") == "exp8_sphor/exp8_sphor_dinov2_base"
    assert hf_exp_dir("exp4_sphor_ncd_convnext_tiny") == "exp9_ncd/exp9_ncd_convnext_tiny"
    assert hf_exp_dir("exp5_sphor_ncd_resnet50") == "exp5_sphor_ncd/exp5_sphor_ncd_resnet50"


def test_adapter_path():
    base = hf_exp_dir("exp2_supcon_dinov2_base")
    assert base == "exp7_supcon/exp7_supcon_dinov2_base"
    assert f"{base}/best_fold_1_adapter" == ("exp7_supcon/exp7_supcon_dinov2_base/best_fold_1_adapter")


def test_unknown_returns_none():
    assert hf_exp_dir("exp99_unknown") is None
    assert hf_file("exp99_unknown", "x.pt") is None
