from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def letterbox_resize(image, target_width, target_height):
    """
    Resizes image maintaining aspect ratio and adds padding (letterbox) to reach target size.
    Prevents both distortion and information loss.
    """
    iw, ih = image.size
    w, h = target_width, target_height
    
    # Calculate scale while preserving aspect ratio
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Resize image with high quality Lanczos filter
    image = image.resize((nw, nh), Image.Resampling.LANCZOS)

    # Create black canvas and paste resized image in the center
    new_image = Image.new('RGB', (w, h), (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image

def resize_image(image, target_width, target_height):
    """Alias for letterbox_resize to maintain compatibility."""
    return letterbox_resize(image, target_width, target_height)

def is_correct_image_size(image, target_width, target_height):
    width, height = image.size
    return width == target_width and height == target_height

class ResizeImageTransform:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        logging.debug(f'Applying letterbox resize transform to image of size {image.size}')
        if not is_correct_image_size(image, self.width, self.height):
            image = letterbox_resize(image, self.width, self.height)
        return image