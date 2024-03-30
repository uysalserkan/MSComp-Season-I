from pykuwahara import kuwahara
import numpy as np
import albumentations as A


def median_ep(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Median edge preserving filter with albumentations library.
    :param img: Image as numpy array.
    :param kernel_size: Kernel size for median filter, 5 or 15.
    :return: Median filtered image as numpy array.
    """""
    ep_img = A.augmentations.median_blur(img.astype(np.uint8), kernel_size)

    return ep_img


def kuwahara_ep(img: np.ndarray, kernel_size: int, method: str) -> np.ndarray:
    """
    Kuwahara edge preserving filter with pykuwahara library.
    :param img: Image as numpy array.
    :param kernel_size: Kernel size for Kuwahara filter, 5 or 15.
    :param method: Method for Kuwahara filter, `mean` or `gaussian`.
    :return: Kuwahara filtered image as numpy array.
    """""
    ep_img = kuwahara(img.astype(np.uint8), method=method, radius=kernel_size)

    return ep_img
