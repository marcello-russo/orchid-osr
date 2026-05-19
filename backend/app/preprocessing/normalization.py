import torch


def get_mean():
    from app.config import settings
    return settings.get_norm_mean()


def get_std():
    from app.config import settings
    return settings.get_norm_std()


def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    denormalized_image = image * std + mean
    return torch.clamp(denormalized_image, 0, 1)
