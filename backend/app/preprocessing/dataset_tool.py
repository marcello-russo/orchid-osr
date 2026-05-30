import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
from dotenv import dotenv_values
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import app.preprocessing.resizing as resizing
import app.preprocessing.rotation as rotation

config = dotenv_values(".env")


# Used during preprocessing to save data to a file while preserving file names
class TensorDatasetWithName(Dataset):
    def __init__(self, tensors, file_paths, classes=None):
        assert len(tensors[0]) == len(file_paths), "Mismatch between data and file paths length"
        self.tensors = tensors
        self.file_names = [os.path.basename(path) for path in file_paths]
        self.classes = classes if classes is not None else []

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (self.file_names[index],)

    def __len__(self):
        return len(self.tensors[0])


# Loads images from a specific path
# Extends ImageFolder to include file names
# Images must be in folders with the following structure:
# path
# ├── class1
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── class2
# │   ├── img1.jpg
# │   └── ...
# └── ...
class ImageFolderWithName(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithName, self).__getitem__(index)
        path, _ = self.samples[index]
        file_name = os.path.basename(path)
        return original_tuple[0], original_tuple[1], file_name


def getDatasetFromFile(file_path):
    processed_data_dict = torch.load(file_path, weights_only=False)
    processed_dataset = processed_data_dict["data"]
    return processed_dataset


def preprocessImage(image, width, height):
    if rotation.is_landscape(image):
        image = rotation.rotate_image(image)
    if not resizing.is_correct_image_size(image, width, height):
        image = resizing.resize_image(image, width, height)
    return image


def augmentDataPath(data_path, output_path, max_images_per_class, width, height, augmentation_functions):
    """Augments images and saves them to output_path."""
    class_count = 0
    for function in augmentation_functions:
        for idx, foto in enumerate(os.listdir(data_path)):
            if class_count >= max_images_per_class and function != rotation.identity:
                return
            img = Image.open(os.path.join(data_path, foto))
            img = preprocessImage(img, width, height)
            original_name = os.path.splitext(foto)[0]
            output_file_path = os.path.join(output_path, f"{function.__name__}_{original_name}_{idx}.jpg")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            img = function(img)
            img.save(output_file_path)
            class_count += 1


def CalculateBalancedTargets(raw_data_path, class_names, augmentation_functions):
    """
    Calculates balanced target images per class.
    Target = min_class_size * num_augmentations (all classes reach the same number)
    """
    class_sizes = {}
    for class_name in class_names:
        class_path = os.path.join(raw_data_path, class_name)
        if os.path.exists(class_path):
            num_files = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
            class_sizes[class_name] = num_files
        else:
            class_sizes[class_name] = 0

    min_size = min(class_sizes.values()) if class_sizes else 0
    target_per_class = min_size * len(augmentation_functions)

    return target_per_class, class_sizes


def _process_single_image(args):
    """Processes a single image (for parallelization)."""
    data_path, foto, output_path, func_name, idx, width, height = args
    img = Image.open(os.path.join(data_path, foto))
    img = preprocessImage(img, width, height)
    original_name = os.path.splitext(foto)[0]
    output_file_path = os.path.join(output_path, f"{func_name}_{original_name}_{idx}.jpg")

    # Map function name -> function
    func_map = {
        "identity": rotation.identity,
        "horizontal_flip": rotation.horizontal_flip,
        "vertical_flip": rotation.vertical_flip,
        "horizontal_and_vertical_flip": rotation.horizontal_and_vertical_flip,
    }
    func = func_map.get(func_name, rotation.identity)
    img = func(img)
    img.save(output_file_path)
    return True


def augmentDataPathBalanced(
    data_path, output_path, target_images, width, height, augmentation_functions, current_count
):
    """
    Augments images until target_images is reached.
    Parallelized with ProcessPoolExecutor.
    If originals > target: undersample to target.
    If originals < target: oversample with augmentation.
    """
    # Clean output directory first to avoid mixing old and new images
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(data_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    num_workers = min(multiprocessing.cpu_count(), 4)  # Reduced for stability

    # If we have more originals than target, undersample
    if len(images) > target_images:
        images = images[:target_images]

    # Copy originals with identity (parallelized)
    identity_args = [(data_path, foto, output_path, "identity", idx, width, height) for idx, foto in enumerate(images)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_process_single_image, identity_args))

    count = len(images)

    # If more images needed, apply additional augmentations
    needed = target_images - count
    if needed <= 0:
        return count

    aug_idx = 1  # Skip identity (already applied)

    while count < target_images and aug_idx < len(augmentation_functions):
        func = augmentation_functions[aug_idx]
        func_name = func.__name__

        images_needed = min(needed, len(images))
        aug_args = [
            (data_path, images[idx % len(images)], output_path, func_name, idx, width, height)
            for idx in range(count, count + images_needed)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_single_image, aug_args))

        count += images_needed
        needed = target_images - count
        aug_idx += 1

    return count
