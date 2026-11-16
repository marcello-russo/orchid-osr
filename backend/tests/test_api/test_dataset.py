from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.auth import create_jwt

client = TestClient(app)

AUTH_HEADER = {"Authorization": f"Bearer {create_jwt('test', 'user')}"}

MOCK_LISTING = {
    "classes": {
        "O. exaltata": {"train": 40, "test": 10},
        "O. garganica": {"train": 30, "test": 5},
        "O. incubacea": {"train": 50, "test": 15},
    },
    "files": {
        "train": {
            "O. exaltata": ["img_0.jpg", "img_1.jpg", "img_2.jpg"],
            "O. garganica": ["img_0.jpg", "img_1.jpg"],
        },
        "test": {
            "O. exaltata": ["img_0.jpg"],
        },
    },
    "total_images": 100,
}


class TestListClasses:
    def test_returns_200(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            resp = client.get("/api/dataset/classes", headers=AUTH_HEADER)
            assert resp.status_code == 200

    def test_response_structure(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get("/api/dataset/classes", headers=AUTH_HEADER).json()
            assert "classes" in data
            assert "total_train" in data
            assert "total_test" in data

    def test_counts_are_correct(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get("/api/dataset/classes", headers=AUTH_HEADER).json()
            assert data["total_train"] == 120
            assert data["total_test"] == 30
            assert len(data["classes"]) == 3

    def test_item_fields(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            cls = client.get("/api/dataset/classes", headers=AUTH_HEADER).json()["classes"][0]
            assert "name" in cls
            assert "train_count" in cls
            assert "test_count" in cls
            assert "total" in cls

    def test_sorted_alphabetically(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            names = [c["name"] for c in client.get("/api/dataset/classes", headers=AUTH_HEADER).json()["classes"]]
            assert names == sorted(names)

    def test_empty_listing_returns_empty(self):
        empty = {"classes": {}, "files": {"train": {}, "test": {}}, "total_images": 0}
        with (
            patch("app.api.routes.dataset._get_listing", return_value=empty),
            patch("app.api.routes.dataset._get_dataset_base", return_value=Path("/tmp/nonexistent_xyz")),
        ):
            data = client.get("/api/dataset/classes", headers=AUTH_HEADER).json()
            assert data["classes"] == []
            assert data["total_train"] == 0
            assert data["total_test"] == 0

    def test_fallback_filesystem(self, tmp_path):
        base = tmp_path / "cropped_256x512"
        (base / "train" / "O. exaltata").mkdir(parents=True)
        (base / "train" / "O. exaltata" / "a.jpg").write_bytes(b"1")
        (base / "test" / "O. garganica").mkdir(parents=True)
        (base / "test" / "O. garganica" / "b.jpg").write_bytes(b"2")

        empty = {"classes": {}, "files": {"train": {}, "test": {}}, "total_images": 0}
        with (
            patch("app.api.routes.dataset._get_listing", return_value=empty),
            patch("app.api.routes.dataset._get_dataset_base", return_value=base),
        ):
            data = client.get("/api/dataset/classes", headers=AUTH_HEADER).json()
            assert len(data["classes"]) == 2
            assert data["total_train"] + data["total_test"] == 2


class TestListClassImages:
    def test_returns_200(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            resp = client.get("/api/dataset/classes/O. exaltata/images", headers=AUTH_HEADER)
            assert resp.status_code == 200

    def test_response_structure(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get("/api/dataset/classes/O. exaltata/images", headers=AUTH_HEADER).json()
            assert "class_name" in data
            assert "split" in data
            assert "total" in data
            assert "images" in data
            assert data["class_name"] == "O. exaltata"
            assert data["split"] == "train"

    def test_returns_correct_count(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get("/api/dataset/classes/O. exaltata/images", headers=AUTH_HEADER).json()
            assert data["total"] == 3

    def test_pagination(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get(
                "/api/dataset/classes/O. exaltata/images",
                headers=AUTH_HEADER,
                params={"limit": 2, "offset": 0},
            ).json()
            assert len(data["images"]) == 2
            assert data["total"] == 3

    def test_image_fields(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            img = client.get(
                "/api/dataset/classes/O. exaltata/images",
                headers=AUTH_HEADER,
            ).json()["images"][0]
            assert "filename" in img
            assert "url" in img
            assert img["url"].startswith("/api/dataset/images/")

    def test_invalid_split_returns_400(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            resp = client.get(
                "/api/dataset/classes/O. exaltata/images",
                headers=AUTH_HEADER,
                params={"split": "invalid"},
            )
            assert resp.status_code == 400

    def test_test_split(self):
        with patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING):
            data = client.get(
                "/api/dataset/classes/O. exaltata/images",
                headers=AUTH_HEADER,
                params={"split": "test"},
            ).json()
            assert data["split"] == "test"
            assert data["total"] == 1

    def test_class_with_no_images(self):
        with (
            patch("app.api.routes.dataset._get_listing", return_value=MOCK_LISTING),
            patch("app.api.routes.dataset._get_dataset_base", return_value=Path("/tmp/nonexistent_xyz")),
        ):
            data = client.get("/api/dataset/classes/O. incubacea/images", headers=AUTH_HEADER).json()
            assert data["total"] == 0
            assert data["images"] == []

    def test_fallback_filesystem(self, tmp_path):
        base = tmp_path / "cropped_256x512"
        class_dir = base / "train" / "O. exaltata"
        class_dir.mkdir(parents=True)
        (class_dir / "a.jpg").write_bytes(b"1")
        (class_dir / "b.jpg").write_bytes(b"1")

        empty = {"classes": {}, "files": {"train": {}, "test": {}}, "total_images": 0}
        with (
            patch("app.api.routes.dataset._get_listing", return_value=empty),
            patch("app.api.routes.dataset._get_dataset_base", return_value=base),
        ):
            data = client.get("/api/dataset/classes/O. exaltata/images", headers=AUTH_HEADER).json()
            assert data["total"] == 2
            assert len(data["images"]) == 2


class TestThumbnailAtomicWrite:
    """La miniatura non deve essere servita mentre è in scrittura.

    Scrivendo in place, una `FileResponse` concorrente dichiarava un
    `Content-Length` su un file incompleto e poi ne leggeva più byte
    ("Too much data for declared Content-Length") -> 520 a valle.
    """

    def test_write_bytes_atomic_replaces_without_temp_files(self, tmp_path):
        from app.api.routes.dataset import _write_bytes_atomic

        target = tmp_path / "img.webp"
        target.write_bytes(b"vecchio")
        _write_bytes_atomic(target, b"contenuto-nuovo")
        assert target.read_bytes() == b"contenuto-nuovo"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["img.webp"]

    def test_thumbnail_is_webp_and_leaves_no_temp(self, tmp_path):
        from PIL import Image

        base = tmp_path / "cropped_256x512"
        class_dir = base / "train" / "O. exaltata"
        class_dir.mkdir(parents=True)
        Image.new("RGB", (80, 40), (120, 40, 90)).save(class_dir / "a.jpg", "JPEG")

        with patch("app.api.routes.dataset._get_dataset_base", return_value=base):
            resp = client.get(
                "/api/dataset/images/train/O. exaltata/a.jpg",
                headers=AUTH_HEADER,
                params={"size": 300},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/webp"
            assert resp.content[:4] == b"RIFF"

        thumbs = list((tmp_path / ".thumbnails" / "300").rglob("*"))
        assert not [p for p in thumbs if p.name.endswith(".tmp")]
        assert [p.name for p in thumbs if p.suffix == ".webp"] == ["a.webp"]

    def test_repeated_requests_are_identical(self, tmp_path):
        from PIL import Image

        base = tmp_path / "cropped_256x512"
        class_dir = base / "train" / "O. exaltata"
        class_dir.mkdir(parents=True)
        Image.new("RGB", (60, 60), (10, 10, 10)).save(class_dir / "b.jpg", "JPEG")

        with patch("app.api.routes.dataset._get_dataset_base", return_value=base):
            url = "/api/dataset/images/train/O. exaltata/b.jpg"
            first = client.get(url, headers=AUTH_HEADER, params={"size": 300})
            second = client.get(url, headers=AUTH_HEADER, params={"size": 300})
        assert first.status_code == second.status_code == 200
        assert first.content == second.content
