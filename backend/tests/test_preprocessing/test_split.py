import hashlib
import json

import numpy as np
import pytest


class TestMD5:
    def test_returns_hexdigest(self, tmp_path):
        from app.preprocessing.split import _md5

        f = tmp_path / "img.jpg"
        f.write_text("hello")
        expected = hashlib.md5(b"hello").hexdigest()
        assert _md5(f) == expected

    def test_different_files_different_hashes(self, tmp_path):
        from app.preprocessing.split import _md5

        a = tmp_path / "a.jpg"
        a.write_text("abc")
        b = tmp_path / "b.jpg"
        b.write_text("def")
        assert _md5(a) != _md5(b)


class TestListImages:
    def test_returns_empty_for_missing_dir(self):
        from app.preprocessing.split import _list_images

        assert _list_images("/nonexistent/path") == []

    def test_lists_only_images(self, tmp_path):
        from app.preprocessing.split import _list_images

        (tmp_path / "a.jpg").write_text("")
        (tmp_path / "b.jpeg").write_text("")
        (tmp_path / "c.png").write_text("")
        (tmp_path / "d.txt").write_text("")
        (tmp_path / "e.JPG").write_text("")
        result = _list_images(tmp_path)
        assert result == ["a.jpg", "b.jpeg", "c.png", "e.JPG"]


class TestSplitClass:
    def test_returns_train_and_test(self, tmp_path):
        from app.preprocessing.split import _split_class

        for i in range(20):
            (tmp_path / f"img_{i}.jpg").write_text(str(i))
        rng = np.random.default_rng(42)
        train, test = _split_class(tmp_path, 5, rng)
        assert len(train) == 15
        assert len(test) == 5
        assert set(train) | set(test) == {f"img_{i}.jpg" for i in range(20)}

    def test_raises_if_not_enough_images(self, tmp_path):
        from app.preprocessing.split import _split_class

        (tmp_path / "img_0.jpg").write_text("")
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Not enough"):
            _split_class(tmp_path, 5, rng)


class TestCreateCleanSplit:
    def test_private_only_split(self, tmp_path):
        from app.preprocessing.split import create_clean_split

        cropped = tmp_path / "cropped"
        out = tmp_path / "output"
        for cls in ["cls_a", "cls_b"]:
            (cropped / cls).mkdir(parents=True)
            for i in range(20):
                (cropped / cls / f"{i}.jpg").write_text(f"{cls}_{i}")

        manifest = create_clean_split(
            cropped_dir=cropped,
            internet_dir=None,
            output_dir=out,
            class_names=["cls_a", "cls_b"],
            internet_classes=[],
            test_per_class=5,
            seed=42,
        )

        assert manifest["total_train"] == 30
        assert manifest["total_test"] == 10
        assert manifest["md5_overlap"] == 0
        assert (out / "train" / "cls_a").exists()
        assert (out / "test" / "cls_b").exists()
        assert len(list((out / "train" / "cls_a").iterdir())) == 15
        assert len(list((out / "test" / "cls_a").iterdir())) == 5
        assert (out / "split_manifest.json").exists()

    def test_internet_class_split(self, tmp_path):
        from app.preprocessing.split import create_clean_split

        cropped = tmp_path / "cropped"
        internet = tmp_path / "internet"
        out = tmp_path / "output"
        cls = "O.exaltata"
        (cropped / cls).mkdir(parents=True)
        (internet / f"{cls}_Int").mkdir(parents=True)
        for i in range(20):
            (cropped / cls / f"{i}.jpg").write_text(f"priv_{i}")
        for i in range(10):
            (internet / f"{cls}_Int" / f"inet_{i}.jpg").write_text(f"inet_{i}")

        manifest = create_clean_split(
            cropped_dir=cropped,
            internet_dir=internet,
            output_dir=out,
            class_names=[cls],
            internet_classes=[cls],
            test_per_class=5,
            seed=42,
        )

        assert manifest["total_test"] == 5
        assert manifest["md5_overlap"] == 0
        assert (out / "test" / cls).exists()
        assert (out / "train" / cls).exists()

    def test_raises_missing_internet_dir(self, tmp_path):
        from app.preprocessing.split import create_clean_split

        cropped = tmp_path / "cropped"
        internet = tmp_path / "internet"
        internet.mkdir()
        out = tmp_path / "output"
        (cropped / "cls_a").mkdir(parents=True)
        (cropped / "cls_a" / "0.jpg").write_text("data")

        with pytest.raises(FileNotFoundError, match="Internet directory not found"):
            create_clean_split(
                cropped_dir=cropped,
                internet_dir=internet,
                output_dir=out,
                class_names=["cls_a"],
                internet_classes=["cls_a"],
                test_per_class=1,
                seed=42,
            )

    def test_manifest_json_structure(self, tmp_path):
        from app.preprocessing.split import create_clean_split

        cropped = tmp_path / "cropped"
        out = tmp_path / "output"
        (cropped / "cls_a").mkdir(parents=True)
        for i in range(20):
            (cropped / "cls_a" / f"{i}.jpg").write_text(str(i))

        create_clean_split(
            cropped_dir=cropped,
            internet_dir=None,
            output_dir=out,
            class_names=["cls_a"],
            internet_classes=[],
            test_per_class=3,
            seed=42,
        )

        with open(out / "split_manifest.json") as f:
            data = json.load(f)
        assert "seed" in data
        assert "timestamp" in data
        assert "classes" in data
        assert "total_train" in data
        assert "total_test" in data
        assert data["total_train"] == 17
        assert data["total_test"] == 3

    def test_md5_dedup_removes_duplicates(self, tmp_path):
        from app.preprocessing.split import create_clean_split

        cropped = tmp_path / "cropped"
        internet = tmp_path / "internet"
        out = tmp_path / "output"
        cls = "cls_a"
        (cropped / cls).mkdir(parents=True)
        (internet / f"{cls}_Int").mkdir(parents=True)
        for i in range(10):
            (cropped / cls / f"{i}.jpg").write_text(f"same_content_{i}")
        for i in range(10):
            (internet / f"{cls}_Int" / f"inet_{i}.jpg").write_text(f"same_content_{i}")

        manifest = create_clean_split(
            cropped_dir=cropped,
            internet_dir=internet,
            output_dir=out,
            class_names=[cls],
            internet_classes=[cls],
            test_per_class=3,
            seed=42,
        )

        assert manifest["md5_overlap"] == 0
