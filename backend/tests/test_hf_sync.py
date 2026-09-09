"""Tests for app.hf.sync (filters + sync flow, no network)."""

import os


class _StubApi:
    def __init__(self, files):
        self._files = files
        self.downloaded = []

    def list_repo_files(self, repo):
        return list(self._files)

    def get_paths_info(self, repo, path):
        return None


def test_is_essential_filters():
    from app.hf.sync import _is_essential

    assert _is_essential("exp1_ce_resnet18/results.json")
    assert _is_essential("exp1_ce_resnet18/embeddings/labels.npy")
    assert not _is_essential("old/exp1/results.json")
    assert not _is_essential("exp1/ncd_figures/plot.png")
    # best-fold serving weights + adapters ARE essential (classify + test means)
    assert _is_essential("exp1/best_fold_0.pt")
    assert _is_essential("exp1/best_fold_0_adapter/adapter_model.safetensors")
    assert not _is_essential("exp1/last_ckpt.pt")


def test_sync_experiments_downloads_essential(tmp_path, monkeypatch):
    from app.hf import sync as sync_mod
    from app.hf.sync import sync_experiments

    monkeypatch.setattr(sync_mod, "download_file", lambda api, repo, src, dst, token: open(dst, "w").write("{}"))
    monkeypatch.setattr(sync_mod, "list_repo_files", lambda api, repo: api.list_repo_files(repo))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_for_test")
    api = _StubApi(["exp1_ce/exp1_ce_resnet18/results.json", "exp1_ce/exp1_ce_resnet18/best_fold_0.pt"])
    out = sync_experiments(dst=str(tmp_path), repo="dummy/repo", api=api)
    assert out["synced"] == 2
    assert out["errors"] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "exp1_ce_resnet18", "results.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "_index.json"))


def test_sync_experiments_no_token_skips(monkeypatch):
    import app.hf.sync as sync_mod

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sync_mod.settings, "hf_token", "")
    out = sync_mod.sync_experiments(dst="/nonexistent-xyz", repo="dummy/repo")
    assert out == {"synced": 0, "skipped": 0, "errors": 0}
