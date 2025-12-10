import os
from argparse import ArgumentParser

import maccel
import numpy as np

from postprocess import YoloPostProcessAnchorless, scale_boxes
from preprocess import preprocess_yolo
from visualize import YoloVisualizer

if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with compiled model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./yolov9m_global8.mxq",
        help="Path to the compiled MXQ model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./traffic_jam.jpg",
        help="Path to the input image of size 1080x1920",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_global8.jpg",
        help="Path to the output image",
    )
    parser.add_argument(
        "--nms_conf", type=float, default=0.25, help="Confidence threshold for NMS"
    )
    parser.add_argument(
        "--nms_iou", type=float, default=0.7, help="IoU threshold for NMS"
    )

    args = parser.parse_args()

    acc = maccel.Accelerator()
    mc = maccel.ModelConfig()
    mc.set_global8_core_mode()
    model = maccel.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPostProcessAnchorless(args.nms_conf, args.nms_iou)
    visualizer = YoloVisualizer()

    img = preprocess_yolo(args.image_path, img_size=(640, 640))
    img = np.transpose(img, [2, 0, 1])  # shape: (3, 640, 640)
    outputs = model.infer([img])
    result = postprocess(outputs)
    result[0] = scale_boxes((640, 640), result[0], (1080, 1920))

    visualizer.save(result, input_path=args.image_path, output_path=args.output_path)
    model.dispose()
