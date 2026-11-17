# tests/test_preprocessing/test_resize_dataset.py
"""Resize di dataset interi: struttura delle classi, rotazione, errori.

`resize_dataset` è il passo che materializza `dataset/cropped_256x512` e
`cropped_224x224` per il training. Un errore silenzioso qui (immagine saltata,
classe persa, file non-immagine copiato) si vede solo come F1 più bassa.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.preprocessing import resize_dataset as rd


def _save(path: Path, size: tuple[int, int], color=(120, 40, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class TestResizeDataset:
    def test_letterboxes_portrait_images_per_class(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        _save(src / "ophrys_lutea" / "a.png", (40, 80))
        _save(src / "ophrys_lutea" / "b.png", (80, 40))  # landscape → ruotata
        _save(src / "ophrys_apifera" / "c.png", (30, 30))

        summary = rd.resize_dataset(src, dst, 64, 32)

        assert summary == {"total": 3, "resized": 3, "errors": 0}
        for name in ("a.png", "b.png"):
            with Image.open(dst / "ophrys_lutea" / name) as img:
                assert img.size == (64, 32)
                assert img.mode == "RGB"
        with Image.open(dst / "ophrys_apifera" / "c.png") as img:
            assert img.size == (64, 32)
        # struttura preservata, nessun file in più
        assert sorted(p.name for p in dst.iterdir()) == ["ophrys_apifera", "ophrys_lutea"]

    def test_landscape_is_rotated_before_letterbox(self, tmp_path):
        """Un'immagine 100×10 diventa verticale: nel letterbox verticale occupa
        tutta l'altezza, cioè i pixel non neri sono più di quanti ne avrebbe
        lasciati la versione orizzontale (letterboxata ai lati)."""
        src = tmp_path / "in"
        dst = tmp_path / "out"
        _save(src / "cls" / "wide.png", (100, 10), color=(255, 255, 255))

        rd.resize_dataset(src, dst, 32, 64)

        with Image.open(dst / "cls" / "wide.png") as img:
            assert img.size == (32, 64)
            non_black = int((np.asarray(img.convert("L")) > 10).sum())
        # Ruotata (10×100) e scalata a 6×64 → ~6*64 = 384 pixel chiari.
        # Senza rotazione sarebbe 32×3 = ~96.
        assert non_black > 300

    def test_non_image_files_are_ignored(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        _save(src / "cls" / "ok.jpg", (20, 20))
        (src / "cls" / "notes.txt").write_text("non un'immagine")
        (src / "cls" / "manifest.json").write_text("{}")

        summary = rd.resize_dataset(src, dst, 16, 16)

        assert summary["total"] == 1
        assert sorted(p.name for p in (dst / "cls").iterdir()) == ["ok.jpg"]

    def test_corrupt_image_is_counted_as_error(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        _save(src / "cls" / "good.png", (20, 20))
        (src / "cls" / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")

        summary = rd.resize_dataset(src, dst, 16, 16)

        assert summary == {"total": 2, "resized": 1, "errors": 1}
        assert (dst / "cls" / "good.png").exists()
        assert not (dst / "cls" / "broken.jpg").exists()

    def test_empty_input_returns_zero_summary(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        assert rd.resize_dataset(src, tmp_path / "out", 16, 16) == {"total": 0, "resized": 0, "errors": 0}

    def test_loose_files_in_root_are_not_classes(self, tmp_path):
        """Solo le sottodirectory sono classi: un'immagine nella radice non deve
        finire in una classe inventata."""
        src = tmp_path / "in"
        _save(src / "stray.png", (20, 20))
        _save(src / "cls" / "a.png", (20, 20))

        summary = rd.resize_dataset(src, tmp_path / "out", 16, 16)

        assert summary["total"] == 1
        assert not (tmp_path / "out" / "stray.png").exists()

    def test_output_dir_is_created_recursively(self, tmp_path):
        src = tmp_path / "in"
        _save(src / "cls" / "a.png", (20, 20))
        dst = tmp_path / "deep" / "nested" / "out"
        rd.resize_dataset(src, dst, 16, 16)
        assert (dst / "cls" / "a.png").exists()

    def test_rerun_overwrites_without_duplicating(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        _save(src / "cls" / "a.png", (20, 20))
        rd.resize_dataset(src, dst, 16, 16)
        rd.resize_dataset(src, dst, 32, 32)
        assert sorted(p.name for p in (dst / "cls").iterdir()) == ["a.png"]
        with Image.open(dst / "cls" / "a.png") as img:
            assert img.size == (32, 32)


class TestResizeForAllArchitectures:
    def test_one_directory_per_registered_size(self, tmp_path, monkeypatch):
        src = tmp_path / "cropped" / "train"
        _save(src / "cls" / "a.png", (20, 20))
        base = tmp_path / "dataset"

        monkeypatch.setattr(
            "app.core.models.registry.unique_input_sizes",
            lambda: {"256x512": (256, 512), "224x224": (224, 224)},
        )

        result = rd.resize_for_all_architectures(src, base)

        assert set(result) == {"256x512", "224x224"}
        assert result["256x512"] == str(base / "cropped_256x512")
        with Image.open(base / "cropped_256x512" / "cls" / "a.png") as img:
            assert img.size == (256, 512)
        with Image.open(base / "cropped_224x224" / "cls" / "a.png") as img:
            assert img.size == (224, 224)

    def test_split_name_nests_output(self, tmp_path, monkeypatch):
        src = tmp_path / "cropped" / "train"
        _save(src / "cls" / "a.png", (20, 20))

        monkeypatch.setattr(
            "app.core.models.registry.unique_input_sizes",
            lambda: {"224x224": (224, 224)},
        )

        result = rd.resize_for_all_architectures(src, tmp_path / "dataset", split_name="train")

        assert result["224x224"] == str(tmp_path / "dataset" / "cropped_224x224" / "train")
        assert (tmp_path / "dataset" / "cropped_224x224" / "train" / "cls" / "a.png").exists()

    @pytest.mark.parametrize("split", [None, "test"])
    def test_returns_only_labels(self, tmp_path, monkeypatch, split):
        monkeypatch.setattr(
            "app.core.models.registry.unique_input_sizes",
            lambda: {"64x64": (64, 64)},
        )
        src = tmp_path / "in"
        _save(src / "cls" / "a.png", (20, 20))
        result = rd.resize_for_all_architectures(src, tmp_path / "base", split_name=split)
        assert list(result) == ["64x64"]
