import numpy as np


def uniform_noise(img: np.ndarray, low: int = 0, high: int = 255, perc: float = 0.1) -> np.ndarray:
    """
    Add uniform noise to the image.
    :param img: Image as numpy array.
    :param low: Uniform noise values low.
    :param high: Uniform noise values high.
    :param perc: Noise apply percentage.
    :return: blurred image as numpy array
    """
    noise = np.random.uniform(low=low, high=high, size=img.shape[:2])

    noised_pixels = np.random.rand(*img.shape[:2]) < perc
    mask = np.where(noised_pixels, noise, 0)

    blurred_img = img + mask

    return blurred_img