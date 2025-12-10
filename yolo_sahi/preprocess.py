import os
from multiprocessing.pool import ThreadPool
from typing import List, Union

import cv2
import numpy as np

NUM_THREADS = min(16, max(1, os.cpu_count() - 1))


def patch_division(img_path: Union[str, np.ndarray]):
    """
    Patch the division of the image to the size of 640x640 and two large patches of size 1080x1080.
    Assume that the input is a path to the image of size 1080x1920 or the image itself.
    """
    if isinstance(img_path, str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = img_path

    assert (
        img.shape[0] == 1080 and img.shape[1] == 1920 and img.shape[2] == 3
    ), "The input image must be of size 1080x1920x3."

    p0 = img[:640, :640]  # start with [0,0]
    p1 = img[440:, :640]  # start with [440,0]
    p2 = img[:640, 640:1280]  # start with [0,640]
    p3 = img[440:, 640:1280]  # start with [440,640]
    p4 = img[:640, 1280:1920]  # start with [0,1280]
    p5 = img[440:, 1280:1920]  # start with [440,1280]
    p6 = img[220:860, 320:960]  # start with [220,320]
    p7 = img[220:860, 960:1600]  # start with [220,960]

    return [p0, p1, p2, p3, p4, p5, p6, p7]


def preprocess_yolo(img_path: Union[str, np.ndarray], img_size=(640, 640)):
    if isinstance(img_path, str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = img_path

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # orig hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # resize the longer side to the target size
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # wh padding
    if dw != 0 or dh != 0:  # if padding is needed, add border to the image
        dw /= 2  # divide padding into 2 sides
        dh /= 2  # to center the image
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

    img = (img / 255).astype(np.float32)

    return img


def convert_fhd_to_batch(img_path: Union[str, np.ndarray]):
    if isinstance(img_path, str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        img = img_path

    assert (
        img.shape[0] == 1080 and img.shape[1] == 1920 and img.shape[2] == 3
    ), "The input image must be of size 1080x1920x3."

    patches = patch_division(img)

    with ThreadPool(NUM_THREADS) as pool:
        preprocessed_patches = pool.map(preprocess_yolo, patches)

    return np.stack(preprocessed_patches, axis=0)  # shape: (8, 640, 640, 3)
