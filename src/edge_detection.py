from __future__ import annotations
import cv2
import numpy as np

def _grayscale(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray=_grayscale(image)
    sx=cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    sy=cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(sx,sy))

def canny_edges(image: np.ndarray, lower_threshold:int=10, upper_threshold:int=35) -> np.ndarray:
    return cv2.Canny(_grayscale(image), lower_threshold, upper_threshold)
