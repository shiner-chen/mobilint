from argparse import ArgumentParser

import torch
from qubee import mxq_compile

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--inference_scheme",
        type=str,
        default="multi",
        choices=["multi", "global8"],
        help="Inference scheme to use",
    )
    args = parser.parse_args()

    onnx_path = "./yolov9m.onnx"
    calib_data_path = "./yolov9m_cali"
    save_path = f"./yolov9m_{args.inference_scheme}.mxq"

    mxq_compile(
        model=onnx_path,
        calib_data_path=calib_data_path,
        quantize_method="maxpercentile",  # quantization method to use
        is_quant_ch=True,  # whether to use channel-wise quantization
        quantize_percentile=0.999,
        topk_ratio=0.01,
        quant_output="ch",  # quantization method for the output layer
        save_path=save_path,
        optimize_option=2,  # optmize option for Aries
        inference_scheme=args.inference_scheme,
        backend="onnx",
        device="gpu" if torch.cuda.is_available() else "cpu",
    )
