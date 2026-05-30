# tests/test_core/test_models.py
import pytest
import torch

from app.core.models.base import ClassifierModel, ModelConfig
from app.core.models.resnet import ResNetClassifier, ResNetConfig


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()

    assert config.num_classes == 6
    assert config.pretrained is True


def test_model_config_custom():
    """Test ModelConfig with custom values."""
    config = ModelConfig(num_classes=10, pretrained=False)

    assert config.num_classes == 10
    assert config.pretrained is False


def test_classifier_model_is_abstract():
    """Test that ClassifierModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ClassifierModel(ModelConfig())


class TestResNetClassifier:
    """Tests for ResNetClassifier implementation."""

    def test_resnet_config_defaults(self):
        """Test ResNetConfig default values."""
        config = ResNetConfig()

        assert config.variant == "resnet18"
        assert config.num_classes == 6
        assert config.pretrained is True

    def test_resnet_config_resnet50(self):
        """Test ResNetConfig with resnet50 variant."""
        config = ResNetConfig(variant="resnet50")

        assert config.variant == "resnet50"

    def test_resnet_forward_shape(self, tiny_dataloader, device):
        """Test that forward pass produces correct output shape."""
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, y, _ in tiny_dataloader:
            x = x.to(device)
            output = model(x)

            assert output.shape == (x.shape[0], 2)
            break

    def test_resnet_embedding_shape(self, tiny_dataloader, device):
        """Test that get_embeddings produces correct output shape."""
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, y, _ in tiny_dataloader:
            x = x.to(device)
            embeddings = model.get_embeddings(x)

            assert embeddings.shape[0] == x.shape[0]
            assert embeddings.shape[1] == model.embedding_dim
            break

    def test_resnet_model_name(self):
        """Test model_name property."""
        config18 = ResNetConfig(variant="resnet18")
        config50 = ResNetConfig(variant="resnet50")

        model18 = ResNetClassifier(config18)
        model50 = ResNetClassifier(config50)

        assert model18.model_name == "resnet18"
        assert model50.model_name == "resnet50"

    def test_resnet_from_config(self):
        """Test from_config factory method."""
        config = ResNetConfig(num_classes=10, pretrained=False)
        model = ResNetClassifier.from_config(config)

        assert model.num_classes == 10

    def test_resnet_save_load_weights(self, tmp_path, device):
        """Test saving and loading model weights."""
        config = ResNetConfig(num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        # Save weights
        weights_path = tmp_path / "test_weights.pt"
        model.save_weights(str(weights_path))

        # Create new model and load weights
        model2 = ResNetClassifier(config).to(device)
        model2.load_weights(str(weights_path))

        # Verify weights are identical
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(param1, param2), f"Parameters {name1} differ"

    def test_resnet50_embedding_dimension(self, tiny_dataloader, device):
        """ResNet50 should produce embedding_dim-dim embeddings."""
        config = ResNetConfig(variant="resnet50", num_classes=2, pretrained=False)
        model = ResNetClassifier(config).to(device)

        for x, _, _ in tiny_dataloader:
            x = x.to(device)
            embeddings = model.get_embeddings(x)
            assert embeddings.shape[1] == model.embedding_dim
            break
