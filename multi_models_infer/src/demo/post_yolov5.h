#ifndef DEMO_INCLUDE_POST_YOLOV5_H_
#define DEMO_INCLUDE_POST_YOLOV5_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "post.h"

struct YOLOv5PostItem {
    uint64_t id;
    cv::Mat& im;
    std::vector<std::vector<float>>& npu_outs;
    std::vector<std::array<float, 4>>& boxes;
    std::vector<float>& scores;
    std::vector<int>& labels;
    std::vector<std::vector<float>>& extras;
};

class YOLOv5PostProcessor : public PostProcessor {
public:
    YOLOv5PostProcessor(int nl, int nc, int no, int imh, int imw, float conf_thres,
                        float iou_thres, bool verbose);
    ~YOLOv5PostProcessor();

public:
    void generate_grids(int imh, int imw, std::vector<int> strides);
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    float sigmoid(float num);
    float inverse_sigmoid(float num);
    virtual std::vector<float> get_extra(const std::vector<float>& output,
                                         const std::vector<int>& grid,
                                         const std::vector<int>& anchor, int stride,
                                         int idx, int grid_idx, float conf_value = 0);
    virtual int get_cls_offset();
    float area(float xmin, float ymin, float xmax, float ymax);
    float get_iou(const std::array<float, 4>& box1, const std::array<float, 4>& box2);
    void xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes);
    void decode_conf_thres(const std::vector<float>& npu_out,
                           const std::vector<int>& grid,
                           const std::vector<std::vector<int>>& anchor, int stride,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<float>& pred_conf, std::vector<int>& pred_label,
                           std::vector<std::pair<float, int>>& pred_scores,
                           std::vector<std::vector<float>>& pred_extra);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_conf, const std::vector<int>& pred_label,
             std::vector<std::pair<float, int>>& scores,
             const std::vector<std::vector<float>>& pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra);
    double set_timer();

    std::vector<std::array<float, 4>>& get_result_box();
    std::vector<float>& get_result_score();
    std::vector<int>& get_result_label();
    std::vector<std::vector<float>>& get_result_extra();
    void compute_ratio_pads(const cv::Mat& im, float& ratio, float& xpad, float& ypad);
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels);
    virtual void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras);

    void print(std::string msg);
    virtual void worker();
    uint64_t enqueue(cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels, std::vector<std::vector<float>>& extras);
    void receive(uint64_t receipt_no);

protected:
    int m_no;            // number outputs per anchor (5 + nc + keypoints/landmarks/masks)
    int m_nextra;        // number of keypoints/landmarks/masks
    int m_nl;            // number of detection layers
    int m_nc;            // number of classes
    int m_na;            // number of anchors
    uint32_t m_imh;      // model input image height
    uint32_t m_imw;      // model input image width
    float m_conf_thres;  // confidence threshold, used in decoding
    float m_inverse_conf_thres;
    float m_iou_thres;  // iou threshold, used in nms
    bool m_verbose;
    int m_max_det_num;
    bool m_only_person = false;

    const int mOpenmpThreadCount = 2;

    std::vector<std::array<float, 4>> final_boxes;
    std::vector<float> final_scores;
    std::vector<int> final_labels;
    std::vector<std::vector<float>> final_extra;  // keypoints/landmarks/masks

    std::vector<std::vector<int>> m_grids;
    std::vector<std::vector<std::vector<int>>> m_anchors;
    std::vector<int> m_strides;

    std::thread mThread;
    std::queue<YOLOv5PostItem> mQueueIn;
    std::vector<uint64_t> mOut;
    uint64_t ticket;

    std::mutex mPrintMutex;
    std::mutex mMutexIn;
    std::mutex mMutexOut;
    std::condition_variable mCondIn;
    std::condition_variable mCondOut;
    bool destroyed;

    // not the best practice, should find better way
    const std::vector<std::string> COCO_LABELS = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush",
    };

    const std::vector<std::array<int, 3>> COLORS = {{255, 255, 102},
                                                    // {240, 96, 153},
                                                    {151, 157, 255},
                                                    {31, 112, 255},
                                                    {29, 178, 255},
                                                    {49, 210, 207},
                                                    {10, 249, 72},
                                                    {23, 204, 146},
                                                    {134, 219, 61},
                                                    {52, 147, 26},
                                                    {187, 212, 0},
                                                    {168, 153, 44},
                                                    {255, 194, 0},
                                                    {147, 69, 52},
                                                    {255, 115, 100},
                                                    {236, 24, 0},
                                                    {255, 56, 132},
                                                    {133, 0, 82},
                                                    {255, 56, 203},
                                                    {200, 149, 255},
                                                    {199, 55, 255}};
};

#endif