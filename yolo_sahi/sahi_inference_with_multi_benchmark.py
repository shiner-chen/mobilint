import os
import time
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
    parser.add_argument("--nms_conf", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.7)
    parser.add_argument("--nmm_ios", type=float, default=0.5)

    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")

    args = parser.parse_args()

    # ----------------------------------
    # Setup maccel
    # ----------------------------------
    acc = maccel.Accelerator()
    mc = maccel.ModelConfig()
    mc.set_multi_core_mode([maccel.Cluster.Cluster0, maccel.Cluster.Cluster1])
    model = maccel.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPostProcessAnchorless(args.nms_conf, args.nms_iou)
    visualizer = YoloVisualizer()

    # ==================================
    # Load & preprocess input once
    # ==================================
    t_pre0 = time.time()
    img = convert_fhd_to_batch(args.image_path)
    img = np.transpose(img, [0, 3, 1, 2])  # shape: (8, 3, 640, 640)
    t_pre1 = time.time()
    preprocess_ms = (t_pre1 - t_pre0) * 1000

    # ==================================
    # Warmup
    # ==================================
    print(f"\n🔥 Warmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        _ = model.infer([img])

    # ==================================
    # Benchmark Runs
    # ==================================
    infer_times = []
    full_times = []

    print(f"\n🚀 Benchmarking {args.runs} runs...\n")
    for i in range(args.runs):
        t0 = time.time()
        
        # Inference
        t_inf0 = time.time()
        outputs = model.infer([img])
        t_inf1 = time.time()
        
        # Postprocess
        result = postprocess(outputs)
        result = aggregate_results(result)
        result = non_max_merge(result, args.nmm_ios)

        t1 = time.time()
        
        infer_times.append((t_inf1 - t_inf0) * 1000)
        full_times.append((t1 - t0) * 1000)

        print(f"Run {i+1}/{args.runs} → infer: {infer_times[-1]:.2f} ms | total: {full_times[-1]:.2f} ms")

    # Save one result for visualization
    visualizer.save([result], input_path=args.image_path, output_path=args.output_path)

    # ==================================
    # Statistics
    # ==================================
    avg_infer = sum(infer_times) / len(infer_times)
    min_infer = min(infer_times)
    max_infer = max(infer_times)

    avg_total = sum(full_times) / len(full_times)

    fps = 1000.0 / avg_infer  # per-frame inference FPS

    print("\n================== Benchmark Result ==================")
    print(f"Preprocess one-time         : {preprocess_ms:.2f} ms\n")

    print(f"Inference (mean)            : {avg_infer:.2f} ms")
    print(f"Inference (min)             : {min_infer:.2f} ms")
    print(f"Inference (max)             : {max_infer:.2f} ms")
    print(f"Inference FPS               : {fps:.2f} FPS\n")

    print(f"Total pipeline (mean)       : {avg_total:.2f} ms")
    print("=======================================================\n")

    model.dispose()