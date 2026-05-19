# tests/test_api/test_species.py
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def _make_fake_dataset():
    """Create a temporary dataset directory with 2 species."""
    tmp = tempfile.mkdtemp()
    for sp, count in [("O. exaltata", 5), ("O. garganica", 3)]:
        sp_dir = Path(tmp) / sp
        sp_dir.mkdir()
        for i in range(count):
            (sp_dir / f"img_{i}.jpg").write_bytes(b"\x89PNG" + b"\x00" * 100)
    return Path(tmp)


class TestSpeciesEndpoint:

    def test_species_returns_200(self):
        fake = _make_fake_dataset()
        with patch("app.api.routes.species._get_dataset_dir", return_value=fake):
            response = client.get("/api/species")
            assert response.status_code == 200

    def test_species_response_structure(self):
        fake = _make_fake_dataset()
        with patch("app.api.routes.species._get_dataset_dir", return_value=fake):
            data = client.get("/api/species").json()

            assert "species" in data
            assert "total_images" in data
            assert "total_species" in data
            assert len(data["species"]) == 2

    def test_species_items_have_fields(self):
        fake = _make_fake_dataset()
        with patch("app.api.routes.species._get_dataset_dir", return_value=fake):
            data = client.get("/api/species").json()
            first = data["species"][0]

            assert "name" in first
            assert "image_count" in first
            assert first["image_count"] == 5

    def test_species_totals_consistent(self):
        fake = _make_fake_dataset()
        with patch("app.api.routes.species._get_dataset_dir", return_value=fake):
            data = client.get("/api/species").json()

            assert data["total_species"] == 2
            assert data["total_images"] == 8
            assert data["total_images"] == sum(s["image_count"] for s in data["species"])

    def test_species_sorted_alphabetically(self):
        fake = _make_fake_dataset()
        with patch("app.api.routes.species._get_dataset_dir", return_value=fake):
            data = client.get("/api/species").json()
            names = [s["name"] for s in data["species"]]

            assert names == sorted(names)
