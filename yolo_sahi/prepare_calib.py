import cv2
import numpy as np
from qubee.calibration import make_calib_man

from preprocess import preprocess_yolo

if __name__ == "__main__":
    DATA_DIR = "./COCO_Num100/"
    SAVE_DIR = "./"
    SAVE_NAME = "yolov9m_cali"
    MAX_SIZE = 100

    make_calib_man(
        pre_ftn=preprocess_yolo,  # callable function to pre-process the calibration data
        data_dir=DATA_DIR,  # path to folder of original calibration data files such as images
        save_dir=SAVE_DIR,  # path to folder to save pre-processed calibration data files
        save_name=SAVE_NAME,  # tag for the generated calibration dataset
        seed=42,  # seed for random selection of calibration data
        max_size=MAX_SIZE,  # Maximum number of data to use for calibration
    )
