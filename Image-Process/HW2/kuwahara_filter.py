from pykuwahara import kuwahara
import numpy as np


def kuwahara_EP(img: np.ndarray, kernel_size: int, method: str) -> np.ndarray:
    """docstring..."""
    ep_img = kuwahara(img.astype(np.uint8), method=method, radius=kernel_size)

    return ep_img