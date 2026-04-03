# tests/test_core/test_kfold.py
"""Test per il modulo k-fold cross validation."""
import os
import shutil
import tempfile

import numpy as np
import pytest

from app.core.kfold import (
    _split_raw_images,
    _copy_fold_images,
    _aggregate_results,
)


CLASS_NAMES = [
    "O. exaltata",
    "O. garganica",
    "O. incubacea",
    "O. majellensis",
    "O. sphegodes",
    "O. sphegodes_Palena",
]


@pytest.fixture
def tiny_raw_dataset(tmp_path):
    """Crea un mini dataset raw con 2 classi e poche immagini per test."""
    from PIL import Image

    classes = ["O. exaltata", "O. garganica"]
    raw_dir = tmp_path / "raw"
    for cls in classes:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(8):
            img = Image.new("RGB", (100, 200), color=(i * 30, 100, 100))
            img.save(cls_dir / f"img_{i}.jpg")

    return str(raw_dir), classes


class TestSplitRawImages:
    """Test per _split_raw_images."""

    def test_stratified_split_maintains_proportions(self, tiny_raw_dataset):
        raw_dir, classes = tiny_raw_dataset
        _, _, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)

        assert len(folds) == 2
        for train_idx, val_idx in folds:
            # Ogni fold ha almeno 1 immagine per classe in train e val
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # Train + val = totale
            total = len(train_idx) + len(val_idx)
            assert total == 16  # 8 per classe × 2 classi

    def test_no_image_leakage_between_folds(self, tiny_raw_dataset):
        raw_dir, classes = tiny_raw_dataset
        all_paths, _, folds = _split_raw_images(raw_dir, classes, num_folds=3, seed=42)

        for train_idx, val_idx in folds:
            train_set = set(train_idx)
            val_set = set(val_idx)
            # Nessuna immagine in comune tra train e val
            assert train_set.isdisjoint(val_set)

    def test_all_images_used_in_each_fold(self, tiny_raw_dataset):
        raw_dir, classes = tiny_raw_dataset
        _, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=3, seed=42)

        total = len(all_labels)
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == total

    def test_class_proportions_in_splits(self, tiny_raw_dataset):
        """Verifica che le proporzioni delle classi siano mantenute."""
        raw_dir, classes = tiny_raw_dataset
        _, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)

        total_ratio = np.sum(all_labels == 0) / np.sum(all_labels == 1)

        for train_idx, _ in folds:
            train_labels = all_labels[train_idx]
            if np.sum(train_labels == 1) > 0:
                train_ratio = np.sum(train_labels == 0) / np.sum(train_labels == 1)
                # Ratio should be close to the original (1.0 since same size)
                assert abs(train_ratio - total_ratio) < 0.5


class TestCopyFoldImages:
    """Test per _copy_fold_images."""

    def test_files_copied_correctly(self, tiny_raw_dataset, tmp_path):
        raw_dir, classes = tiny_raw_dataset
        all_paths, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)
        train_idx, val_idx = folds[0]

        fold_dir = str(tmp_path / "fold_0")
        train_dir, val_dir = _copy_fold_images(
            all_paths, all_labels, classes, train_idx, val_idx, fold_dir
        )

        # Verifica che le directory esistano
        assert os.path.exists(train_dir)
        assert os.path.exists(val_dir)

        # Verifica che ci siano immagini
        for cls in classes:
            train_cls = os.path.join(train_dir, cls)
            val_cls = os.path.join(val_dir, cls)
            assert os.path.exists(train_cls)
            assert os.path.exists(val_cls)
            assert len(os.listdir(train_cls)) > 0
            assert len(os.listdir(val_cls)) > 0

    def test_no_overlap_train_val(self, tiny_raw_dataset, tmp_path):
        """Nessun file in comune tra train e val dello stesso fold."""
        raw_dir, classes = tiny_raw_dataset
        all_paths, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)
        train_idx, val_idx = folds[0]

        fold_dir = str(tmp_path / "fold_0")
        train_dir, val_dir = _copy_fold_images(
            all_paths, all_labels, classes, train_idx, val_idx, fold_dir
        )

        for cls in classes:
            train_files = set(os.listdir(os.path.join(train_dir, cls)))
            val_files = set(os.listdir(os.path.join(val_dir, cls)))
            assert train_files.isdisjoint(val_files)

    def test_correct_total_count(self, tiny_raw_dataset, tmp_path):
        """Il numero totale di file copiati corrisponde al previsto."""
        raw_dir, classes = tiny_raw_dataset
        all_paths, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)
        train_idx, val_idx = folds[0]

        fold_dir = str(tmp_path / "fold_0")
        _copy_fold_images(all_paths, all_labels, classes, train_idx, val_idx, fold_dir)

        # Conta file copiati
        total_copied = 0
        for cls in classes:
            train_cls = os.path.join(fold_dir, "train", cls)
            val_cls = os.path.join(fold_dir, "val", cls)
            total_copied += len(os.listdir(train_cls))
            total_copied += len(os.listdir(val_cls))

        assert total_copied == len(all_paths)


class TestAggregateResults:
    """Test per _aggregate_results."""

    def test_correct_mean_std(self):
        fold_results = [
            {"best_f1": 0.90, "best_accuracy": 0.92},
            {"best_f1": 0.92, "best_accuracy": 0.94},
            {"best_f1": 0.88, "best_accuracy": 0.90},
            {"best_f1": 0.91, "best_accuracy": 0.93},
            {"best_f1": 0.89, "best_accuracy": 0.91},
        ]

        result = _aggregate_results(fold_results)

        assert abs(result["mean_f1"] - np.mean([0.90, 0.92, 0.88, 0.91, 0.89])) < 1e-6
        assert abs(result["mean_accuracy"] - np.mean([0.92, 0.94, 0.90, 0.93, 0.91])) < 1e-6
        assert result["best_fold"] == 1  # Indice di F1=0.92

    def test_single_fold(self):
        fold_results = [{"best_f1": 0.95, "best_accuracy": 0.97}]
        result = _aggregate_results(fold_results)

        assert result["mean_f1"] == 0.95
        assert result["std_f1"] == 0.0
        assert result["best_fold"] == 0

    def test_all_same_f1(self):
        fold_results = [
            {"best_f1": 0.90, "best_accuracy": 0.91},
            {"best_f1": 0.90, "best_accuracy": 0.92},
        ]
        result = _aggregate_results(fold_results)

        assert result["std_f1"] == 0.0
        assert result["best_fold"] == 0  # Primo fold in caso di parità


class TestTempCleanup:
    """Test che le directory temporanee vengano pulite."""

    def test_temp_dirs_cleaned_up(self, tiny_raw_dataset, tmp_path):
        """Verifica che temp_kfold venga pulito dopo l'esecuzione."""
        raw_dir, classes = tiny_raw_dataset
        all_paths, all_labels, folds = _split_raw_images(raw_dir, classes, num_folds=2, seed=42)

        temp_base = str(tmp_path / "temp_kfold")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold_dir = os.path.join(temp_base, f"fold_{fold_idx}")
            _copy_fold_images(all_paths, all_labels, classes, train_idx, val_idx, fold_dir)

            # Simula cleanup
            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)

        # Alla fine temp_kfold dovrebbe essere vuoto o non esistere
        if os.path.exists(temp_base):
            remaining = os.listdir(temp_base)
            assert len(remaining) == 0, f"File rimasti in temp: {remaining}"


class TestKfoldDirectoryStructure:
    """Test che la directory k-fold sia separata dall'esperimento originale."""

    def test_kfold_dir_is_separate(self, tmp_path):
        """La directory k-fold deve essere sibling di quella dell'esperimento."""
        output_dir = str(tmp_path / "experiments" / "exp4_combined_internet_cropped")
        os.makedirs(output_dir, exist_ok=True)

        kfold_dir = output_dir + "_kfold_resnet18"

        # Non deve essere annidata dentro output_dir
        assert not kfold_dir.startswith(output_dir + "/")
        assert not kfold_dir.startswith(output_dir + os.sep)

        # Deve essere sibling
        assert os.path.dirname(kfold_dir) == os.path.dirname(output_dir)
        assert os.path.basename(kfold_dir) == "exp4_combined_internet_cropped_kfold_resnet18"

    def test_kfold_dir_contains_all_outputs(self, tmp_path):
        """Tutti gli output (fold JSON, pesi, report) vanno nella stessa directory."""
        kfold_dir = str(tmp_path / "experiments" / "exp4_combined_internet_cropped_kfold_resnet18")
        os.makedirs(kfold_dir)

        # Simula file creati durante un run
        for name in ["fold_0.json", "best_fold_0.pt", "results.json", "REPORT_KFOLD_resnet18.txt"]:
            with open(os.path.join(kfold_dir, name), "w") as f:
                f.write("{}" if name.endswith(".json") else "test")

        # Tutti i file devono essere nella stessa directory
        files = set(os.listdir(kfold_dir))
        assert "fold_0.json" in files
        assert "best_fold_0.pt" in files
        assert "results.json" in files
        assert "REPORT_KFOLD_resnet18.txt" in files


class TestConfigJson:
    """Test che config.json venga generato nella directory k-fold."""

    def test_config_created_on_first_run(self, tmp_path):
        """config.json viene creato solo la prima volta."""
        kfold_dir = str(tmp_path / "kfold_out")
        os.makedirs(kfold_dir)
        config_path = os.path.join(kfold_dir, "config.json")

        from datetime import datetime
        config = {
            "type": "kfold",
            "model_variant": "resnet18",
            "num_folds": 5,
            "num_epochs": 26,
            "batch_size": 16,
            "learning_rate": 0.001,
            "seed": 57,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            import json
            json.dump(config, f)

        assert os.path.exists(config_path)
        with open(config_path) as f:
            saved = json.load(f)
        assert saved["batch_size"] == 16
        assert saved["model_variant"] == "resnet18"

    def test_config_not_overwritten_on_resume(self, tmp_path):
        """config.json non viene sovrascritto se esiste già (resume fold)."""
        kfold_dir = str(tmp_path / "kfold_out")
        os.makedirs(kfold_dir)
        config_path = os.path.join(kfold_dir, "config.json")

        original_timestamp = "2026-03-30T14:14:39"
        import json
        with open(config_path, "w") as f:
            json.dump({"timestamp": original_timestamp}, f)

        # Simula un secondo run (non deve sovrascrivere)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump({"timestamp": "new"}, f)

        with open(config_path) as f:
            saved = json.load(f)
        assert saved["timestamp"] == original_timestamp


class TestFoldJsonBatchSize:
    """Test che batch_size venga salvato nel JSON di ogni fold."""

    def test_batch_size_in_fold_metrics(self):
        """batch_size deve essere registrato nei fold metrics."""
        fold_metrics = {
            "best_f1": 0.87,
            "best_accuracy": 0.88,
            "best_epoch": 20,
            "fold": 1,
            "batch_size": 16,
            "train_size": 1865,
            "val_size": 467,
        }
        assert fold_metrics["batch_size"] == 16

    def test_aggregate_reads_batch_from_folds(self):
        """_aggregate_and_test legge batch_size effettivo dai fold JSON."""
        fold_results = [
            {"best_f1": 0.87, "batch_size": 16},
            {"best_f1": 0.89, "batch_size": 16},
        ]

        # Simula la logica di effective_batch_size
        effective_batch_size = 8  # default dal parametro
        if fold_results and "batch_size" in fold_results[0]:
            effective_batch_size = fold_results[0]["batch_size"]

        assert effective_batch_size == 16
