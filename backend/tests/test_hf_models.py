"""Tests for app.hf.models (upload logic, no network)."""

import json
import os


class _StubApi:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, repo_id, path_or_fileobj, path_in_repo):
        assert os.path.exists(path_or_fileobj)
        self.uploaded.append(path_in_repo)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs.get("path_in_repo"))


def _make_exp(base, name, best_fold=0):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"best_fold": best_fold, "mean_f1": 0.9, "test_metrics": {"f1_macro": 0.88}}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"num_classes": 6}, f)
    open(os.path.join(d, f"best_fold_{best_fold}.pt"), "w").write("dummy")
    return d


def test_upload_model_uploads_best_fold(tmp_path):
    from app.hf.models import upload_model

    _make_exp(str(tmp_path), "exp5_sphor_ncd_resnet18")
    api = _StubApi()
    assert upload_model(api, "resnet18", "exp5_sphor_ncd", "exp5_sphor_ncd", exp_base=str(tmp_path)) is True
    assert "exp5_sphor_ncd/exp5_sphor_ncd_resnet18/results.json" in api.uploaded
    assert "exp5_sphor_ncd/exp5_sphor_ncd_resnet18/best_fold_0.pt" in api.uploaded


def test_upload_model_skips_without_results(tmp_path):
    from app.hf.models import upload_model

    api = _StubApi()
    assert upload_model(api, "resnet18", "exp5_sphor_ncd", "exp5_sphor_ncd", exp_base=str(tmp_path)) is False
    assert api.uploaded == []


class _StubNcdApi:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, repo_id, path_in_repo, path_or_fileobj):
        self.uploaded.append(path_in_repo)


def test_upload_ncd_uploads_json_and_figures(tmp_path):
    from app.hf.ncd import upload_ncd

    exp = tmp_path / "exp5_sphor_ncd_resnet18"
    (exp / "ncd_figures").mkdir(parents=True)
    (exp / "ncd_results.json").write_text("{}")
    (exp / "ncd_figures" / "plot.png").write_text("x")
    api = _StubNcdApi()
    n = upload_ncd(api, exp_base=str(tmp_path))
    assert n == 2
    assert "exp5_sphor_ncd/exp5_sphor_ncd_resnet18/ncd_results.json" in api.uploaded
    assert "exp5_sphor_ncd/exp5_sphor_ncd_resnet18/ncd_figures/plot.png" in api.uploaded


def test_upload_ncd_empty_dir_returns_zero(tmp_path):
    from app.hf.ncd import upload_ncd

    api = _StubNcdApi()
    assert upload_ncd(api, exp_base=str(tmp_path)) == 0
    assert api.uploaded == []
