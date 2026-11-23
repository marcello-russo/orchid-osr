from collections.abc import Sequence

import torch


def get_mean() -> list[float]:
    from app.config import settings

    return settings.get_norm_mean()


def get_std() -> list[float]:
    from app.config import settings

    return settings.get_norm_std()


def denormalize_image(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    denormalized_image = image * std_t + mean_t
    return torch.clamp(denormalized_image, 0, 1)
