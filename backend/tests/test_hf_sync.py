"""Tests for app.hf.sync (filters + sync flow, no network)."""

import os
import types


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


class _StubApiSized:
    """Stub con dimensioni remote, per testare lo skip e il mismatch."""

    def __init__(self, files, sizes):
        self._files = files
        self._sizes = sizes

    def list_repo_files(self, repo):
        return list(self._files)

    def get_paths_info(self, repo, paths):
        if isinstance(paths, str):
            paths = [paths]
        return [types.SimpleNamespace(path=p, size=self._sizes[p]) for p in paths if p in self._sizes]


def test_sync_skips_files_with_matching_size(tmp_path, monkeypatch):
    from app.hf import sync as sync_mod

    src = "exp1_ce/exp1_ce_resnet18/results.json"
    target = tmp_path / "exp1_ce_resnet18" / "results.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12345")

    downloaded = []
    monkeypatch.setattr(sync_mod, "download_file", lambda *a, **k: downloaded.append(a))
    monkeypatch.setattr(sync_mod, "list_repo_files", lambda api, repo: api.list_repo_files(repo))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_for_test")

    api = _StubApiSized([src], {src: 5})
    out = sync_mod.sync_experiments(dst=str(tmp_path), repo="dummy/repo", api=api)

    assert out == {"synced": 0, "skipped": 1, "errors": 0}
    assert downloaded == []


def test_sync_detects_size_mismatch(tmp_path, monkeypatch):
    from app.hf import sync as sync_mod

    src = "exp1_ce/exp1_ce_resnet18/results.json"
    monkeypatch.setattr(
        sync_mod,
        "download_file",
        lambda api, repo, src_path, dst, token: open(dst, "wb").write(b"x"),
    )
    monkeypatch.setattr(sync_mod, "list_repo_files", lambda api, repo: api.list_repo_files(repo))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_for_test")

    api = _StubApiSized([src], {src: 10})
    out = sync_mod.sync_experiments(dst=str(tmp_path), repo="dummy/repo", api=api)

    assert out["errors"] == 1
    assert out["synced"] == 0


def test_sync_lock_blocks_concurrent_run(tmp_path, monkeypatch):
    import fcntl

    from app.hf import sync as sync_mod

    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_for_test")
    monkeypatch.setattr(sync_mod, "list_repo_files", lambda api, repo: api.list_repo_files(repo))

    lock = open(tmp_path / sync_mod.LOCK_FILENAME, "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        out = sync_mod.sync_experiments(dst=str(tmp_path), repo="dummy/repo", api=_StubApi([]))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    assert out == {"synced": 0, "skipped": 0, "errors": 0}
