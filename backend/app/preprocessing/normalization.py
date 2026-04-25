import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from dotenv import set_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_fresh_mean_std(loader):
    mean = torch.zeros(3)
    squared_sum = torch.zeros(3)
    total_pixels = 0
    total_batches = len(loader)

    for batch_idx, (images, labels, filenames) in enumerate(loader):
        batch_size, channels, height, width = images.shape
        num_pixels_in_batch = batch_size * height * width

        images = images.view(batch_size, channels, -1)

        mean += images.sum([0, 2])
        squared_sum += (images ** 2).sum([0, 2])
        total_pixels += num_pixels_in_batch

        progress = (batch_idx + 1) / total_batches * 100
        logging.info(f'Processing batch {batch_idx + 1}/{total_batches} ({progress:.2f}%)')

    mean /= total_pixels
    std = torch.sqrt(squared_sum / total_pixels - mean ** 2)

    logging.info(f'Mean: {mean}')
    logging.info(f'Mean RGB: {(mean * 255).int().tolist()}')
    logging.info(f'Std: {std}')
    logging.info(f'Std RGB: {(std * 255).int().tolist()}')
    logging.info(f'Total pixels: {total_pixels}')

    # Save to .env (may fail in container with mounted volumes)
    try:
        set_key('.env', 'NORM_MEAN', ' '.join([str(x.item()) for x in mean]))
        set_key('.env', 'NORM_STD', ' '.join([str(x.item()) for x in std]))
    except OSError:
        logging.warning('Could not write to .env file (mounted volume issue)')

    mean = str(mean.tolist())
    std = str(std.tolist())

    return mean, std

def get_mean():
    """Get normalization mean from settings."""
    from app.config import settings
    return settings.get_norm_mean()

def get_std():
    """Get normalization std from settings."""
    from app.config import settings
    return settings.get_norm_std()

def converto_to_rgb(x):
    return (x * 255).int().tolist()

def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    denormalized_image = image * std + mean
    return torch.clamp(denormalized_image, 0, 1)

class NormalizeImageTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return T.Normalize(self.mean, self.std)(image)
