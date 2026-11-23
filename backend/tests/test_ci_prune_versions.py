"""Il prune GHCR cancella solo ciò che è sicuro cancellare.

La funzione `plan` è pura e vive in `scripts/ci/prune_versions.py` (usata dal
workflow `Publish Docker images`). Questi test coprono i casi che, sbagliati,
si pagano in produzione: rollback sparito, manifest figlio cancellato, tag
ripubblicato contato due volte.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "prune_versions.py"
_spec = importlib.util.spec_from_file_location("prune_versions", _PATH)
assert _spec and _spec.loader
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def version(vid: int, tags: list[str], created: str) -> dict:
    return {
        "id": vid,
        "created_at": created,
        "metadata": {"container": {"tags": tags}},
    }


class TestPlan:
    def test_keeps_newest_semver_tags(self):
        versions = [
            version(1, ["v0.2.0", "sha-aaa"], "2026-11-16T15:00:00Z"),
            version(2, ["v0.2.1", "sha-bbb"], "2026-11-16T16:00:00Z"),
            version(3, ["v0.2.2", "sha-ccc"], "2026-11-16T17:00:00Z"),
        ]
        to_delete, keep = prune.plan(versions, keep=3)
        assert keep == {"v0.2.0", "v0.2.1", "v0.2.2"}
        assert to_delete == {}

    def test_deletes_only_outside_keep(self):
        versions = [version(i, [f"v0.1.{i}", f"sha-{i}"], f"2026-11-1{i}T00:00:00Z") for i in range(5)]
        to_delete, keep = prune.plan(versions, keep=2)
        assert keep == {"v0.1.3", "v0.1.4"}
        assert sorted(to_delete) == [0, 1, 2]  # v0.1.0, v0.1.1, v0.1.2

    def test_semver_order_not_creation_order(self):
        """Ripubblicare un tag vecchio non deve scalzare i tag nuovi."""
        versions = [
            version(1, ["v0.2.3", "sha-old"], "2026-11-16T16:00:00Z"),
            version(2, ["v0.2.4", "sha-cur"], "2026-11-16T17:00:00Z"),
            version(3, ["v0.2.2", "sha-new"], "2026-11-17T09:00:00Z"),  # rebuilder, ma semver più basso
        ]
        to_delete, keep = prune.plan(versions, keep=2)
        assert keep == {"v0.2.4", "v0.2.3"}
        assert to_delete == {3: "fuori dalle ultime 2"}

    def test_republished_tag_deletes_stale_duplicate(self):
        """Stesso tag su due record: resta il più recente, l'altro è spazzatura."""
        versions = [
            version(10, ["v0.2.4", "sha-vecchio"], "2026-11-16T17:00:00Z"),
            version(11, ["v0.2.4", "sha-nuovo"], "2026-11-17T09:00:00Z"),
            version(12, ["v0.2.3", "sha-altro"], "2026-11-16T16:00:00Z"),
        ]
        to_delete, keep = prune.plan(versions, keep=3)
        assert keep == {"v0.2.4", "v0.2.3"}
        assert to_delete == {10: "duplicato di un tag ripubblicato"}

    def test_untagged_versions_are_never_deleted(self):
        """I manifest figli dell'indice non devono mai essere toccati."""
        versions = [
            version(1, ["v0.1.0", "sha-aaa"], "2026-11-16T15:00:00Z"),
            version(2, ["v0.1.1", "sha-bbb"], "2026-11-16T16:00:00Z"),
            version(99, [], "2026-11-16T16:00:01Z"),
            version(98, [], "2026-11-16T16:00:02Z"),
        ]
        to_delete, keep = prune.plan(versions, keep=1)
        assert keep == {"v0.1.1"}
        assert to_delete == {1: "fuori dalle ultime 1"}

    def test_sha_only_version_is_left_alone(self):
        versions = [
            version(1, ["v0.1.0", "sha-aaa"], "2026-11-16T15:00:00Z"),
            version(2, ["sha-bbb"], "2026-11-16T16:00:00Z"),
        ]
        to_delete, keep = prune.plan(versions, keep=1)
        assert keep == {"v0.1.0"}
        assert to_delete == {}

    def test_empty_package(self):
        to_delete, keep = prune.plan([], keep=3)
        assert to_delete == {}
        assert keep == set()

    @pytest.mark.parametrize("tag,expected", [("v0.2.10", (0, 2, 10)), ("v1.0.0", (1, 0, 0))])
    def test_semkey(self, tag, expected):
        assert prune.semkey(tag) == expected
