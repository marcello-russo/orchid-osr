import torch
import torchvision.transforms as T
import os
from PIL import Image
import app.preprocessing.rotation as rotation, app.preprocessing.resizing as resizing
from dotenv import dotenv_values
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import numpy as np
import random

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

# Dataset loaded from a single folder
# Images must be directly in the folder
# path
# ├── img1.jpg
# ├── img2.jpg
# └── ...
class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Carica immagine come RGB
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


def getDatasetFromFile(file_path):
    processed_data_dict = torch.load(file_path, weights_only=False)
    processed_dataset = processed_data_dict['data']
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
            output_file_path = os.path.join(output_path, f'{function.__name__}_{original_name}_{idx}.jpg')
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            img = function(img)
            img.save(output_file_path)
            class_count += 1


def SplitData(data_path, output_path, class_name, train_size, validation_size):
    """Splits images into train, validation and test sets."""
    train_path = os.path.join(output_path, 'train', class_name)
    validation_path = os.path.join(output_path, 'validation', class_name)
    test_path = os.path.join(output_path, 'test', class_name)

    # Clean existing directories to avoid accumulating old files
    for path in [train_path, validation_path, test_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    images = os.listdir(os.path.join(data_path, class_name))
    random.shuffle(images)

    for idx, foto in enumerate(images):
        if idx < train_size*len(os.listdir(os.path.join(data_path, class_name))):
            shutil.copy(os.path.join(data_path, class_name, foto), train_path)
        elif idx < (train_size+validation_size)*len(os.listdir(os.path.join(data_path, class_name))):
            shutil.copy(os.path.join(data_path, class_name, foto), validation_path)
        else:
            shutil.copy(os.path.join(data_path, class_name, foto), test_path)

    return train_path, validation_path, test_path


# size: numero di immagini per classe (ignorato se balanced=True)
# raw_data_path: percorso delle immagini originali
# work_processed_data_path_balanced: percorso di lavoro per le immagini augmentate
# out_processed_data_path: percorso di output per le immagini augmentate
# class_names: nomi delle classi
# train_size: percentuale di immagini da mettere nel training set
# validation_size: percentuale di immagini da mettere nel validation set
# width: larghezza delle immagini
# height: altezza delle immagini
# augmentation_functions: funzioni di trasformazione da applicare alle immagini
# balanced: se True, bilancia tutte le classi allo stesso numero di immagini
def DynamicAugmentation(size, raw_data_path, work_processed_data_path_balanced, out_processed_data_path, class_names, train_size, validation_size, width, height, augmentation_functions, balanced=True):
    """
    If balanced=True: all classes are augmented to the same number of images.
    If balanced=False: each class is augmented to its maximum.
    """
    # Clean work directory at start to avoid accumulating old files
    if os.path.exists(work_processed_data_path_balanced):
        shutil.rmtree(work_processed_data_path_balanced)

    if balanced:
        target_per_class, class_sizes = CalculateBalancedTargets(raw_data_path, class_names, augmentation_functions)
        train_target = int(target_per_class * train_size)

    for class_name in class_names:
        print(f'Processing {class_name}...')
        train_path, validation_path, test_path = SplitData(raw_data_path, work_processed_data_path_balanced, class_name, train_size, validation_size)

        # Clean and create output directories
        for split in ['train', 'validation', 'test']:
            out_dir = os.path.join(out_processed_data_path, split, class_name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

        if balanced:
            train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            augmentDataPathBalanced(train_path, os.path.join(out_processed_data_path, 'train', class_name),
                                   train_target, width, height, augmentation_functions, train_images)
        else:
            augmentDataPath(train_path, os.path.join(out_processed_data_path, 'train', class_name), size*train_size, width, height, augmentation_functions)

        # Validation and test: identity only (no additional augmentation)
        augmentDataPath(validation_path, os.path.join(out_processed_data_path, 'validation', class_name), size*0, width, height, [rotation.identity])
        augmentDataPath(test_path, os.path.join(out_processed_data_path, 'test', class_name), size*0, width, height, [rotation.identity])

    return True

def CalculateAugmentationSize(raw_data_path, class_names, augmentation_functions):
    """Calculates min and max images per class."""
    min_size = 1000000
    max_size = 0
    for class_name in class_names:
        class_path = os.path.join(raw_data_path, class_name)
        num_files = len(os.listdir(class_path))
        if num_files < min_size:
            min_size = num_files
        if num_files > max_size:
            max_size = num_files
    return min_size * len(augmentation_functions), max_size * len(augmentation_functions)


def CalculateBalancedTargets(raw_data_path, class_names, augmentation_functions):
    """
    Calculates balanced target images per class.
    Target = min_class_size * num_augmentations (all classes reach the same number)
    """
    class_sizes = {}
    for class_name in class_names:
        class_path = os.path.join(raw_data_path, class_name)
        if os.path.exists(class_path):
            num_files = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
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
    output_file_path = os.path.join(output_path, f'{func_name}_{original_name}_{idx}.jpg')

    # Map function name -> function
    func_map = {
        'identity': rotation.identity,
        'horizontal_flip': rotation.horizontal_flip,
        'vertical_flip': rotation.vertical_flip,
        'horizontal_and_vertical_flip': rotation.horizontal_and_vertical_flip
    }
    func = func_map.get(func_name, rotation.identity)
    img = func(img)
    img.save(output_file_path)
    return True


def augmentDataPathBalanced(data_path, output_path, target_images, width, height, augmentation_functions, current_count):
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

    images = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    num_workers = min(multiprocessing.cpu_count(), 4)  # Reduced for stability

    # If we have more originals than target, undersample
    if len(images) > target_images:
        images = images[:target_images]

    # Copy originals with identity (parallelized)
    identity_args = [(data_path, foto, output_path, 'identity', idx, width, height)
                     for idx, foto in enumerate(images)]

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
        aug_args = [(data_path, images[idx % len(images)], output_path, func_name, idx, width, height)
                    for idx in range(count, count + images_needed)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_single_image, aug_args))

        count += images_needed
        needed = target_images - count
        aug_idx += 1

    return count

