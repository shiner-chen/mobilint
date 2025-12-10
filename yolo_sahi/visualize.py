from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
import torch

from coco import get_coco_det_palette, get_coco_label


def draw_boxes(img, xyxy, desc, cls_color):
    """plot bounding boxes on image"""
    h, w, _ = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(
        img, (x1, y1), (x2, y2), cls_color, thickness=tl, lineType=cv2.LINE_AA
    )

    tf = max(tl - 1, 1)  # font thickness
    cv2.putText(
        img,
        desc,
        (x1, y1 - 2),
        0,
        tl / 2,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


class BaseVisualizer(ABC):
    def __init__(self, dataset: str = "coco"):
        self.dataset = dataset

        if self.dataset == "coco":
            self.get_label = get_coco_label
            self.get_color = get_coco_det_palette
        else:
            raise NotImplementedError(f"Got unsupported dataset: ", self.dataset)

    @abstractmethod
    def save(self, out_post_processed, **kwargs):
        pass


class YoloVisualizer(BaseVisualizer):
    def __init__(self) -> None:
        super().__init__("coco")
        self.model_input_size = [640, 640]

    def save(
        self,
        out_post_processed: List[torch.Tensor],
        input_path: str,
        output_path: str = None,
        masks: List[torch.Tensor] = None,
        kpts: List[torch.Tensor] = None,
    ):
        assert not (
            masks is not None and kpts is not None
        ), "masks and kpts cannot be used together."

        img = cv2.imread(input_path)

        img = self.draw_det(out_post_processed, img)

        if output_path is not None:  # image demo
            cv2.imwrite(output_path, img)

        return img

    def draw_det(
        self,
        out_post_processed: List[torch.Tensor],
        img: np.ndarray,
    ):
        det = out_post_processed[0]
        num_det = det.shape[0]

        for j in range(num_det):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            cls = det[j, 5].cpu().numpy().item()
            cls_name = self.get_label(int(cls))
            cls_color = self.get_color(int(cls))
            desc = f"{cls_name}: {round(100 * conf.item(), 1)}%"
            img = draw_boxes(img, xyxy, desc, cls_color)

        return img
