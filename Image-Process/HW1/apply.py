from argparse import ArgumentParser
from glob import glob
import os
import albumentations as A
from pathlib import Path
import cv2
import numpy as np
from noises import (gaussian_noise, uniform_noise)
from epfs import (median_ep, kuwahara_ep)


IMAGE_INIT_TRANSFORMS = A.Compose([
    # A.transforms.ToGray()  # cv2.COLOR.. aynı işi yapıyor, kaldırıldı.
])
FILTER_SIZE = {"kernel_5": 5, "kernel_15": 15}
NOISE_LEVELS = {"0_1": 0.1, "0_5": 0.5, "0_8": 0.8}
NOISE_METHOD = ["gaussian", "uniform"]
# EP_METHOD = ["median", "kuwahara"]


def read_image(path: str) -> np.ndarray:
    """
    Read image from the given path and apply initial transformations.
    :param path: Path to the image.
    :return: Image as numpy array after initial transformations
    """
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = IMAGE_INIT_TRANSFORMS(image=img)

        return img["image"]

    except Exception as exc:
        print(f"Error -> {exc}")


def operate_on_images(image_paths: list) -> dict:
    """
    Apply operations on the given image list.
    :param image_paths: List of image paths.
    :return: A dictionary of images after operations.
    """
    imgs_dict = {
        "orj": {},
        "gaussian": {
            "0_1": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
            "0_5": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
            "0_8": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
        },
        "uniform": {
            "0_1": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
            "0_5": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
            "0_8": {
                "img": {},
                "kernel_5": {
                    "median": {},
                    "kuwahara": {}
                },
                "kernel_15": {
                    "median": {},
                    "kuwahara": {}
                },
            },
        },
    }

    for each_noise in NOISE_METHOD:
        for noise_level_str, noise_level in NOISE_LEVELS.items():
            for each_img in image_paths:
                tmp_img = read_image(each_img)
                tmp_img_name = each_img.split(os.sep)[-1]

                imgs_dict['orj'][tmp_img_name] = tmp_img

                if each_noise == NOISE_METHOD[0]:
                    noisy_img = gaussian_noise(img=tmp_img, perc=noise_level)
                else:
                    noisy_img = uniform_noise(img=tmp_img, perc=noise_level)

                imgs_dict[each_noise][noise_level_str]["img"][tmp_img_name] = noisy_img

                for filter_level_str, filter_level in FILTER_SIZE.items():
                    median_blur_img = median_ep(img=noisy_img, kernel_size=filter_level)
                    kuwahara_blur_img = kuwahara_ep(img=noisy_img, kernel_size=filter_level, method="mean")

                    imgs_dict[each_noise][noise_level_str][filter_level_str]["median"][tmp_img_name] = median_blur_img
                    imgs_dict[each_noise][noise_level_str][filter_level_str]["kuwahara"][tmp_img_name] = kuwahara_blur_img  # NOQA

    return imgs_dict


def save_images(images: dict) -> None:
    """
    Save images to the disk.
    :param images: Dictionary of images.
    :return: None
    """
    for each_noise in NOISE_METHOD:
        # Path.mkdir(Path(f"{each_noise}"), exist_ok=True, parents=True)
        for noise_level_str, noise_level in NOISE_LEVELS.items():
            for each_img in images[each_noise][noise_level_str]["img"]:
                orj_img = images["orj"][each_img]
                noisy_img = images[each_noise][noise_level_str]["img"][each_img]
                tmp_img_name = each_img.split(".")[0]

                # cv2.imwrite(f"{tmp_img_name}_{each_noise}_{noise_level_str}_img.jpg", tmp_img)

                for filter_level_str, filter_level in FILTER_SIZE.items():
                    median_blur_img = images[each_noise][noise_level_str][filter_level_str]["median"][each_img]
                    kuwahara_blur_img = images[each_noise][noise_level_str][filter_level_str]["kuwahara"][each_img]

                    save_path = f"{each_noise}/{tmp_img_name}/{noise_level_str}/{filter_level_str}/"
                    Path.mkdir(Path(save_path), exist_ok=True, parents=True)

                    cv2.imwrite(f"{save_path}orj.jpg", orj_img)
                    cv2.imwrite(f"{save_path}noisy.jpg", noisy_img)
                    cv2.imwrite(f"{save_path}median.jpg", median_blur_img)  # NOQA
                    cv2.imwrite(f"{save_path}kuwahara.jpg", kuwahara_blur_img)  # NOQA


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--extension", type=str, default="jpg", help="Image extension.")
    parser.add_argument("--path", type=str, default='./', help="Path to the images.")
    args = parser.parse_args()

    img_names = glob(f"{args.path}*.{args.extension}")

    images = operate_on_images(image_paths=img_names)
    save_images(images=images)
