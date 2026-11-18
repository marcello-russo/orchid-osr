# tests/test_api/test_dataset_listing.py
"""La pagina Dataset in produzione vive di un artefatto, non del disco.

Il container non ha il dataset: `classes` e le miniature vengono dal listing
`app/data/dataset_listing.json` spedito con l'immagine, più il proxy CDN di
HuggingFace per i byte. Per giorni il listing è stato l'unica cosa che teneva
in piedi la pagina, senza nessun test: un file sbagliato o assente avrebbe
lasciato la pagina vuota in produzione e verde in CI (l'ho verificato a mano, e
solo perché due endpoint che *scansionano il disco* rispondevano zero ho
creduto che la pagina fosse vuota).

Questi test leggono l'artefatto vero e lo confrontano con `species_counts.json`:
i due file devono raccontare la stessa storia.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app
from app.config import settings
from app.core.auth import create_jwt

LISTING = Path(__file__).resolve().parents[2] / "app" / "data" / "dataset_listing.json"
COUNTS = Path(__file__).resolve().parents[2] / "app" / "data" / "species_counts.json"

# Numeri dello split pulito (seed 57): 2.192 train / 300 test su 6 specie.
TRAIN_TOTAL = 2192
TEST_TOTAL = 300


def _listing() -> dict:
    return json.loads(LISTING.read_text(encoding="utf-8"))


def _auth_headers(role: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_jwt(f'test-{role}', role)}"}


class TestShippedListing:
    def test_listing_covers_the_study_species_with_the_clean_split(self):
        listing = _listing()
        classes = listing["classes"]
        for name in settings.class_names:
            assert name in classes, f"{name} manca dal listing: la pagina non la mostrerebbe"
            assert classes[name]["train"] > 0, f"{name}: zero immagini di training nel listing"
            assert classes[name]["test"] > 0, f"{name}: zero immagini di test nel listing"

        train = sum(classes[n]["train"] for n in settings.class_names)
        test = sum(classes[n]["test"] for n in settings.class_names)
        assert (train, test) == (TRAIN_TOTAL, TEST_TOTAL), f"split nel listing: {train} train / {test} test"

    def test_listing_files_match_the_declared_counts(self):
        """Ogni classe dichiara N immagini e ne elenca N, altrimenti la galleria pagina a vuoto."""
        listing = _listing()
        files = listing["files"]
        for split in ("train", "test"):
            assert split in files, f"manca lo split {split} nel listing"
            for name in settings.class_names:
                listed = files[split].get(name, [])
                assert (
                    len(listed) == listing["classes"][name][split]
                ), f"{name}/{split}: {len(listed)} file elencati, {listing['classes'][name][split]} dichiarati"
                assert len(set(listed)) == len(listed), f"{name}/{split}: nomi duplicati nel listing"

    def test_listing_and_species_counts_agree(self):
        """Due artefatti, una verità: se divergono, una delle due pagine mente."""
        listing = _listing()
        counts = json.loads(COUNTS.read_text(encoding="utf-8"))["species"]
        for name in settings.class_names:
            declared = listing["classes"][name]["train"] + listing["classes"][name]["test"]
            assert counts[name] == declared, f"{name}: species_counts.json dice {counts[name]}, il listing {declared}"


class TestClassesEndpointUsesTheListing:
    def test_returns_exactly_the_study_species(self):
        client = TestClient(app)
        response = client.get("/api/dataset/classes", headers=_auth_headers())
        assert response.status_code == 200
        body = response.json()
        names = [c["name"] for c in body["classes"]]
        # O. sphegodes_Napoli è nel listing (popolazione rimossa) ma non ha file
        # servibili: mostrarla darebbe 548 miniature rotte.
        assert names == sorted(settings.class_names)
        assert "O. sphegodes_Napoli" not in names
        assert body["total_train"] == TRAIN_TOTAL
        assert body["total_test"] == TEST_TOTAL

    def test_lists_filenames_that_exist_in_the_listing(self):
        client = TestClient(app)
        name = settings.class_names[0]
        response = client.get(
            f"/api/dataset/classes/{name}/images",
            params={"split": "train", "limit": 3},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        body = response.json()
        expected = _listing()["files"]["train"][name]
        assert body["total"] == len(expected)
        assert [i["filename"] for i in body["images"]] == expected[:3]
        for image in body["images"]:
            assert image["url"] == f"/api/dataset/images/train/{name}/{image['filename']}"

    def test_the_disk_scanning_endpoints_are_gone(self):
        """`/summary` e `/gallery` scandivano il disco: su un container stateless
        rispondevano zero, senza che niente li chiamasse. Se tornano, devono
        leggere il listing (e avere un test), non il filesystem."""
        client = TestClient(app)
        assert client.get("/api/dataset/summary").status_code == 404
        assert client.get("/api/dataset/gallery", headers=_auth_headers()).status_code == 404
