# tests/test_preprocessing/test_augment.py
import torch
from app.preprocessing.augment import get_augmentation_functions
from app.preprocessing import rotation


class TestGetAugmentationFunctions:
    """Tests for get_augmentation_functions."""

    def test_returns_list(self):
        funcs = get_augmentation_functions()
        assert isinstance(funcs, list)
        assert len(funcs) == 4

    def test_all_callable(self):
        funcs = get_augmentation_functions()
        for f in funcs:
            assert callable(f)


class TestAugmentationEffect:
    """Test that augmentation functions actually transform the image."""

    def test_identity_preserves_image(self):
        """identity() should return the exact same tensor."""
        img = torch.randn(3, 64, 64)
        result = rotation.identity(img)
        assert torch.equal(result, img)

    def test_horizontal_flip_changes_image(self):
        """horizontal_flip should differ from identity for non-symmetric images."""
        img = torch.arange(64 * 64, dtype=torch.float).reshape(1, 64, 64).expand(3, -1, -1)
        flipped = rotation.horizontal_flip(img)
        assert not torch.equal(flipped, img)

    def test_horizontal_flip_is_left_right(self):
        """horizontal_flip should mirror along width dimension."""
        # Create image with distinct left/right halves
        img = torch.zeros(3, 4, 4)
        img[:, :, 0] = 1.0  # left column = 1
        flipped = rotation.horizontal_flip(img)
        assert torch.equal(flipped[:, :, -1], torch.ones(3, 4))  # now right column = 1
        assert torch.equal(flipped[:, :, 0], torch.zeros(3, 4))  # left column = 0

    def test_vertical_flip_changes_image(self):
        """vertical_flip should differ from identity for non-symmetric images."""
        img = torch.arange(64 * 64, dtype=torch.float).reshape(1, 64, 64).expand(3, -1, -1)
        flipped = rotation.vertical_flip(img)
        assert not torch.equal(flipped, img)

    def test_vertical_flip_is_top_bottom(self):
        """vertical_flip should mirror along height dimension."""
        img = torch.zeros(3, 4, 4)
        img[:, 0, :] = 1.0  # top row = 1
        flipped = rotation.vertical_flip(img)
        assert torch.equal(flipped[:, -1, :], torch.ones(3, 4))  # now bottom row = 1
        assert torch.equal(flipped[:, 0, :], torch.zeros(3, 4))  # top row = 0

    def test_double_flip_returns_original(self):
        """horizontal_and_vertical_flip then identity is not original (double flip)."""
        img = torch.randn(3, 64, 64)
        double_flipped = rotation.horizontal_and_vertical_flip(img)
        # Double flip on a non-symmetric image is different from original
        # But double flip = hflip(vflip(img)). Applying hflip again + vflip again = original
        restored = rotation.horizontal_flip(rotation.vertical_flip(double_flipped))
        assert torch.allclose(restored, img)
