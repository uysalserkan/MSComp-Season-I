import cv2
import numpy as np


def cluster_image_segmentation(img: np.ndarray, K: int) -> np.ndarray:
    """docstring"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flatten_img = np.float32(img.reshape((-1, 3)))

    ret,label,center = cv2.kmeans(data=flatten_img, K=K, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2


def OTSU_segmentation(img: np.ndarray) -> np.ndarray:
    """docstring."""
    otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return otsu_img


def graph_based_segmentation(img: np.ndarray, scale: int = 150, sigma: float = 0.85, min_size: int = 250) -> np.ndarray:
    """docstring."""
    segmented_image = cv2.ximgproc.segmentation.createGraphSegmentation()
    segmented_image.setSigma(sigma)
    segmented_image.setK(scale)
    segmented_image.setMinSize(min_size)

    s_img = np.uint8(segmented_image.processImage(img))

    return s_img