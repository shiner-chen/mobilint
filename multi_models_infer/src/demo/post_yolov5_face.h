#ifndef DEMO_INCLUDE_POST_YOLOV5_FACE_H_
#define DEMO_INCLUDE_POST_YOLOV5_FACE_H_

#include "demo/post_yolov5.h"

class YOLOv5FacePostProcessor : public YOLOv5PostProcessor {
public:
    YOLOv5FacePostProcessor(int nl, int nc, int no, int imh, int imw, float conf_thres,
                            float iou_thres, bool verbose);

    std::vector<float> get_extra(const std::vector<float>& output,
                                 const std::vector<int>& grid,
                                 const std::vector<int>& anchor, int stride, int idx,
                                 int grid_idx, float conf_value = 0);
    int get_cls_offset();
    void plot_landmarks(cv::Mat& im, std::vector<std::vector<float>>& landmarks);
    void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras);

    // not the best practice, should find better way
    const std::vector<std::array<int, 3>> LMARK_COLORS = {
        {0, 255, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 255, 0}};
};

#endif