import numpy as np


def gaussian_noise(img: np.ndarray, variance: int = 10, mean: int = 128, perc: float = 0.1) -> np.ndarray:
    """
    Add gaussian noise to the image
    :param img: Image as numpy array.
    :param variance: Gaussian noise values variance.
    :param mean: Gaussian noise values mean.
    :param perc: Noise apply percentage.
    :return: blurred image as numpy array.
    """
    gaussian_noise_np = np.random.normal(mean, variance, img.shape[:2])
    noised_pixels = np.random.rand(*img.shape[:2]) < perc

    mask = np.where(noised_pixels, gaussian_noise_np, 0)
    blurred_img = img + mask

    return blurred_img


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
