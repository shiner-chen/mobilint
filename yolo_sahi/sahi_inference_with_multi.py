import os
from argparse import ArgumentParser

import cv2
import maccel
import numpy as np

from aggregate import aggregate_results, non_max_merge
from postprocess import YoloPostProcessAnchorless
from preprocess import convert_fhd_to_batch
from visualize import YoloVisualizer

if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with compiled model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./yolov9m_multi.mxq",
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
        default="./output_multi.jpg",
        help="Path to the output image",
    )
    parser.add_argument(
        "--nms_conf", type=float, default=0.25, help="Confidence threshold for NMS"
    )
    parser.add_argument(
        "--nms_iou", type=float, default=0.7, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--nmm_ios", type=float, default=0.5, help="IoS threshold for NMM"
    )

    args = parser.parse_args()

    acc = maccel.Accelerator()
    mc = maccel.ModelConfig()
    mc.set_multi_core_mode([maccel.Cluster.Cluster0, maccel.Cluster.Cluster1])
    model = maccel.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPostProcessAnchorless(args.nms_conf, args.nms_iou)
    visualizer = YoloVisualizer()

    img = convert_fhd_to_batch(args.image_path)
    img = np.transpose(img, [0, 3, 1, 2])  # shape: (8, 3, 640, 640)
    outputs = model.infer([img])

    result = postprocess(outputs)
    result = aggregate_results(result)
    result = non_max_merge(result, args.nmm_ios)
    visualizer.save([result], input_path=args.image_path, output_path=args.output_path)
    model.dispose()
