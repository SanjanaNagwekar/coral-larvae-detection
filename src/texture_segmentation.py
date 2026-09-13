from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans

def create_laws_kernels() -> list[np.ndarray]:
    return [
        np.array([1,4,6,4,1],dtype=np.float32),
        np.array([-1,-2,0,2,1],dtype=np.float32),
        np.array([-1,0,2,0,-1],dtype=np.float32),
        np.array([1,-4,6,-4,1],dtype=np.float32),
        np.array([-1,2,0,-2,1],dtype=np.float32),
    ]

def compute_texture_features(grayscale_image: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    source=grayscale_image.astype(np.float32)
    features=[]
    for kernel in kernels:
        h=cv2.filter2D(source, cv2.CV_32F, kernel[np.newaxis,:])
        v=cv2.filter2D(source, cv2.CV_32F, kernel[:,np.newaxis])
        features.append(np.abs(h)+np.abs(v))
    return np.stack(features,axis=-1)

def texture_segmentation(image: np.ndarray, n_clusters:int=5, blur_kernel_size:tuple[int,int]=(15,15), random_state:int=42) -> np.ndarray:
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray, blur_kernel_size,0)
    ft=compute_texture_features(blurred, create_laws_kernels())
    fv=ft.reshape((-1,ft.shape[-1]))
    labels=KMeans(n_clusters=n_clusters,random_state=random_state,n_init=10).fit(fv).labels_.reshape(gray.shape)
    rng=np.random.default_rng(random_state)
    palette=rng.integers(0,256,size=(n_clusters,3),dtype=np.uint8)
    out=np.zeros((*gray.shape,3),dtype=np.uint8)
    for label in range(n_clusters):
        out[labels==label]=palette[label]
    return out
