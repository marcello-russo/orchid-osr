from typing import Any

import torchvision.transforms as T


def rotate_image(image: Any) -> Any:
    return image.rotate(90, expand=True)


def is_landscape(image: Any) -> bool:
    width, height = image.size
    return width > height


def horizontal_flip(img: Any) -> Any:
    return T.functional.hflip(img)


def vertical_flip(img: Any) -> Any:
    return T.functional.vflip(img)


def horizontal_and_vertical_flip(img: Any) -> Any:
    return T.functional.hflip(T.functional.vflip(img))


def identity(img: Any) -> Any:
    return img
