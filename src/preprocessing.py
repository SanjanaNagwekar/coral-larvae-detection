from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image

def gaussian_blur(image: np.ndarray, kernel_size: tuple[int, int]=(15,15), sigma: float=10.0) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, sigma)
