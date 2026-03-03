#include "demo/post_yolov5.h"

YOLOv5PostProcessor::YOLOv5PostProcessor(int nl, int nc, int no, int imh, int imw,
                                         float conf_thres, float iou_thres,
                                         bool verbose) {
    m_nl = nl;  // number of detection layers
    m_nc = nc;  // number of classes
    m_no = no;  // number outputs per anchor (5 + nc + keypts/lmarks/masks)
    m_na = 3;   // number of anchors
    m_nextra = m_no - 5 - m_nc;  // num
    m_imh = imh;                 // model input image height
    m_imw = imw;                 // model input image width
    m_conf_thres = conf_thres;   // confidence threshold, used in decoding
    m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
    m_iou_thres = iou_thres;  // iou threshold, used in nms
    m_verbose = verbose;
    m_max_det_num = 300;

    for (int i = 0; i < m_nl; i++) {
        m_strides.push_back(pow(2, 3 + i));
    }
    generate_grids(imh, imw, m_strides);

    if (m_nl == 3) {
        // YOLOv5 P5 anchors
        m_anchors = {
            {{10, 13}, {16, 30}, {33, 23}},      // P3/8
            {{30, 61}, {62, 45}, {59, 119}},     // P4/16
            {{116, 90}, {156, 198}, {373, 326}}  // P5/32
        };
    } else if (m_nl == 4) {
        // YOLOv5 P6 anchors
        m_anchors = {
            {{19, 27}, {44, 40}, {38, 94}},        // P3/8
            {{96, 68}, {86, 152}, {180, 137}},     // P4/16
            {{140, 301}, {303, 264}, {238, 542}},  // P5/32
            {{436, 615}, {739, 380}, {925, 792}}   // P6/64
        };
    } else {
        throw std::invalid_argument("Number of detection layers should 3 or 4");
    }

    ticket = 0;
    destroyed = false;
    mThread = std::thread(&YOLOv5PostProcessor::worker, this);
}

YOLOv5PostProcessor::~YOLOv5PostProcessor() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();
    mThread.join();
}

/*
        Generate grids
*/
void YOLOv5PostProcessor::generate_grids(int imh, int imw, std::vector<int> strides) {
    for (int i = 0; i < strides.size(); i++) {
        int grid_h = imh / strides[i];
        int grid_w = imw / strides[i];
        int grid_size = grid_h * grid_w * 2;

        std::vector<int> grids;
        for (int j = 0; j < grid_size; j++) {
            if (j % 2 == 0) {
                grids.push_back(((int)j / 2) % grid_w);
            } else {
                grids.push_back(((int)j / 2) / grid_w);
            }
        }

        m_grids.push_back(grids);
    }
}

/*
        Apply sigmoid function to the input float number
*/
float YOLOv5PostProcessor::sigmoid(float num) { return 1 / (1 + exp(-(float)num)); }

float YOLOv5PostProcessor::inverse_sigmoid(float num) { return -log(1 / num - 1); }

/*
        Get extra data out of model outputs. Simple object detection does not
        have any extra data. Face Detection has landmarks, Segmentatin has
        masks and Pose Estimation has keypoints

*/
std::vector<float> YOLOv5PostProcessor::get_extra(const std::vector<float>& output,
                                                  const std::vector<int>& grid,
                                                  const std::vector<int>& anchor,
                                                  int stride, int idx, int grid_idx,
                                                  float conf_value) {
    std::vector<float> extra;
    return extra;
}

/*
        Get the offset for the class values. In Face Detection class values are
        located after landmarks.
*/
int YOLOv5PostProcessor::get_cls_offset() { return 0; }

/*
        Calculate the area of the box
*/
float YOLOv5PostProcessor::area(float xmin, float ymin, float xmax, float ymax) {
    float width = xmax - xmin;
    float height = ymax - ymin;

    if (width < 0) return 0;

    if (height < 0) return 0;

    return width * height;
}

/*
        Calculate IoU of two input boxes
*/
float YOLOv5PostProcessor::get_iou(const std::array<float, 4>& box1,
                                   const std::array<float, 4>& box2) {
    float epsilon = 1e-6;

    // Coordinated of the overlapped region(intersection of two boxes)
    float overlap_xmin = std::max(box1[0], box2[0]);
    float overlap_ymin = std::max(box1[1], box2[1]);
    float overlap_xmax = std::min(box1[2], box2[2]);
    float overlap_ymax = std::min(box1[3], box2[3]);

    // Calculate areas
    float overlap_area = area(overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax);
    float area1 = area(box1[0], box1[1], box1[2], box1[3]);
    float area2 = area(box2[0], box2[1], box2[2], box2[3]);
    float iou = overlap_area / (area1 + area2 - overlap_area + epsilon);

    return iou;
}

/*
        Convert boxes from Center Form(CxCyWiHe) to Corner Form(XminYminXmaxYmax)
*/
void YOLOv5PostProcessor::xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes) {
    for (uint32_t i = 0; i < pred_boxes.size(); i++) {
        float cx = pred_boxes[i][0];
        float cy = pred_boxes[i][1];

        pred_boxes[i][0] = cx - pred_boxes[i][2] * 0.5;
        pred_boxes[i][1] = cy - pred_boxes[i][3] * 0.5;
        pred_boxes[i][2] = cx + pred_boxes[i][2] * 0.5;
        pred_boxes[i][3] = cy + pred_boxes[i][3] * 0.5;
    }
}

/*
        Decoding and masking with conf threshold
*/
void YOLOv5PostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out, const std::vector<int>& grid,
    const std::vector<std::vector<int>>& anchor, int stride,
    std::vector<std::array<float, 4>>& pred_boxes, std::vector<float>& pred_conf,
    std::vector<int>& pred_label, std::vector<std::pair<float, int>>& pred_scores,
    std::vector<std::vector<float>>& pred_extra) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;

    for (int i = 0; i < m_na; i++) {
#pragma omp parallel for num_threads(mOpenmpThreadCount) \
    shared(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra)
        for (int j = 0; j < grid_h * grid_w; j++) {
            int idx = j * m_na * m_no + i * m_no;
            int grid_idx = j * 2;

            if (npu_out[idx + 4] > m_inverse_conf_thres) {
                float conf_value = sigmoid(npu_out[idx + 4]);
                /* decode bounding box coordinates */
                float x =
                    (sigmoid(npu_out[idx + 0]) * 2 - 0.5 + grid[grid_idx + 0]) * stride;
                float y =
                    (sigmoid(npu_out[idx + 1]) * 2 - 0.5 + grid[grid_idx + 1]) * stride;
                float w = pow(sigmoid(npu_out[idx + 2]) * 2, 2) * anchor[i][0];
                float h = pow(sigmoid(npu_out[idx + 3]) * 2, 2) * anchor[i][1];
                std::array<float, 4> temp_box = {x, y, w, h};

                std::vector<float> extra = get_extra(npu_out, grid, anchor[i], stride,
                                                     idx, grid_idx, conf_value);

                for (int k = 0; k < m_nc; k++) {
                    if (m_only_person && k != 0) continue;
                    int cls_offset = get_cls_offset();
                    float cls_score = sigmoid(npu_out[idx + 5 + cls_offset + k]);
                    if (conf_value * cls_score > m_conf_thres) {
#pragma omp critical
                        {
                            pred_conf.push_back(conf_value);
                            pred_label.push_back(k);
                            pred_boxes.push_back(temp_box);
                            pred_scores.push_back(std::make_pair(conf_value * cls_score,
                                                                 pred_scores.size()));
                            pred_extra.push_back(extra);
                        }
                    }
                }
            }
        }
    }
}

/*
        Apply NMS
*/
void YOLOv5PostProcessor::nms(const std::vector<std::array<float, 4>>& pred_boxes,
                              const std::vector<float>& pred_conf,
                              const std::vector<int>& pred_label,
                              std::vector<std::pair<float, int>>& scores,
                              const std::vector<std::vector<float>>& pred_extra,
                              std::vector<std::array<float, 4>>& final_boxes,
                              std::vector<float>& final_scores,
                              std::vector<int>& final_labels,
                              std::vector<std::vector<float>>& final_extra) {
    // sort the scores(predicted confidence * predicted class score)
    sort(scores.begin(), scores.end(), std::greater<>());

    for (int i = 0; i < (int)scores.size(); i++) {
        float temp_score = scores[i].first;
        if (scores[i].first != -99) {  // check if the box valid or not
            int idx = scores[i].second;
            const std::array<float, 4>& max_box = pred_boxes[idx];

            for (int j = i; j < (int)scores.size(); j++) {
                int temp_idx = scores[j].second;
                const std::array<float, 4>& temp_box = pred_boxes[temp_idx];
                float iou = get_iou(max_box, temp_box);

                if (iou > m_iou_thres && pred_label[idx] == pred_label[temp_idx]) {
                    scores[j].first = -99;  // mark the invalid boxes
                }
            }

            final_boxes.push_back(max_box);
            final_scores.push_back(temp_score);
            final_labels.push_back(pred_label[idx]);
            final_extra.push_back(pred_extra[idx]);

            if (final_boxes.size() >= m_max_det_num) {
                break;
            }
        }
    }
}

/*
        Returns current time, used to measure performance time
*/
double YOLOv5PostProcessor::set_timer() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::vector<std::array<float, 4>>& YOLOv5PostProcessor::get_result_box() {
    return final_boxes;
}

std::vector<float>& YOLOv5PostProcessor::get_result_score() { return final_scores; }

std::vector<int>& YOLOv5PostProcessor::get_result_label() { return final_labels; }

std::vector<std::vector<float>>& YOLOv5PostProcessor::get_result_extra() {
    return final_extra;
}

void YOLOv5PostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl) {
        throw std::invalid_argument("Post processing takes " + std::to_string(m_nl) +
                                    " NPU outputs but " +
                                    std::to_string(npu_outs.size()) + " were given");
    }

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_conf;
    std::vector<int> pred_label;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<std::vector<float>> pred_extra;

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[i], m_grids[i], m_anchors[i], m_strides[i], pred_boxes,
                          pred_conf, pred_label, pred_scores, pred_extra);
    }

    xywh2xyxy(pred_boxes);

    nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, final_boxes,
        final_scores, final_labels, final_extra);
    double end = set_timer();
    if (m_verbose) std::cout << "Real C++ Time        : " << end - start << std::endl;
}

/*
        Compute the ratio and pads needed to switch between
        original image size and model input image size
*/
void YOLOv5PostProcessor::compute_ratio_pads(const cv::Mat& im, float& ratio, float& xpad,
                                             float& ypad) {
    cv::Size size = im.size();
    if (size.width > size.height) {
        ratio = (float)m_imw / size.width;
        xpad = 0;
        ypad = (m_imh - ratio * size.height) / 2;
    } else {
        ratio = (float)m_imh / size.height;
        xpad = (m_imw - ratio * size.width) / 2;
        ypad = 0;
    }
}

/*
        Draw detected box and write the it's label & score
*/
void YOLOv5PostProcessor::plot_boxes(cv::Mat& im,
                                     std::vector<std::array<float, 4>>& boxes,
                                     std::vector<float>& scores,
                                     std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        if (m_only_person && labels[i] != 0) continue;
        int xmin = (int)(boxes[i][0] - xpad) / ratio;
        int ymin = (int)(boxes[i][1] - ypad) / ratio;
        int xmax = (int)(boxes[i][2] - xpad) / ratio;
        int ymax = (int)(boxes[i][3] - ypad) / ratio;
        rect.x = xmin;
        rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        std::array<int, 3> bgr = COLORS[labels[i] % 20];
        cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
        cv::rectangle(im, rect, clr, 2);

        double font_scale = std::min(std::max(rect.width / 500.0, 0.35), 0.99);
        // std::string desc = COCO_LABELS[labels[i]] + " " +
        // 	std::to_string((int)(scores[i] * 100)) + "%";
        // cv::putText(im, desc, cv::Point(xmin, ymin-10),
        // 	cv::FONT_HERSHEY_SIMPLEX,
        // 	font_scale, clr, 1, false);
    }
}

/*
        Plot extra data such as landmarks. keypoins, masks.
        Simple Object Detection does not any extras.
*/
void YOLOv5PostProcessor::plot_extras(cv::Mat& im,
                                      std::vector<std::vector<float>>& extras) {
    // do nothing
    int do_nothing;
}

/*
        Print out given message in DEBUG mode using locks
*/
void YOLOv5PostProcessor::print(std::string msg) {
#ifdef DEBUG
    std::lock_guard<std::mutex> lk(mPrintMutex);
    std::cout << msg << endl;
#endif
}

void YOLOv5PostProcessor::worker() {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Worker | ";

    print(title + "Start");
    while (!destroyed) {
        std::unique_lock<std::mutex> lk(mMutexIn);
        if (mQueueIn.empty()) {
            mCondIn.wait(lk, [this] { return !mQueueIn.empty() || destroyed; });
        }

        if (destroyed) {
            break;
        }

        auto k = mQueueIn.front();
        mQueueIn.pop();
        lk.unlock();

        auto start = set_timer();

        run_postprocess(k.npu_outs);
        k.boxes = get_result_box();
        k.scores = get_result_score();
        k.labels = get_result_label();
        k.extras = get_result_extra();

        plot_boxes(k.im, k.boxes, k.scores, k.labels);
        plot_extras(k.im, k.extras);

        auto end = set_timer();
        auto elapsed = std::to_string(end - start);

        print(title + "Postprocessing time: " + elapsed);
        print(title + "Number of detections " + std::to_string(k.boxes.size()));

        std::unique_lock<std::mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        std::unique_lock<std::mutex> lk_(mMutexOut);  // JUST IN CASE
        mCondOut.notify_all();
        lk_.unlock();  // JUST IN CASE
    }
    print(title + "Finish");
}

uint64_t YOLOv5PostProcessor::enqueue(cv::Mat& im,
                                      std::vector<std::vector<float>>& npu_outs,
                                      std::vector<std::array<float, 4>>& boxes,
                                      std::vector<float>& scores,
                                      std::vector<int>& labels,
                                      std::vector<std::vector<float>>& extras) {
    // Note: proto used only in Segmentation, in all other tasks it is NULL

    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Enqueue | ";

    print(title + "Start");
    uint64_t ticket_save = 0;

    {
        std::lock_guard<std::mutex> lk(mMutexIn);
        mQueueIn.push({++ticket, im, npu_outs, boxes, scores, labels, extras});
        ticket_save = ticket;

        mCondIn.notify_all();  // JUST IN CASE
        print(title + "Input Queue size " + std::to_string(mQueueIn.size()));
    }
    print(title + "Finish");
    return ticket_save;
}

void YOLOv5PostProcessor::receive(uint64_t receipt_no) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Receive | ";

    while (!destroyed) {
        print(title + "Start | Receipt = " + std::to_string(receipt_no));
        std::unique_lock<std::mutex> lk(mMutexOut);

        if (mOut.empty()) {
            mCondOut.wait(lk, [this] { return !(mOut.empty()) || destroyed; });
        }
        print(title + "Received Output Queue of size " + std::to_string(mOut.size()));

        if (destroyed) {
            break;
        }

        for (int i = 0; i < mOut.size(); i++) {
            if (mOut[i] == receipt_no) {
                print(title + "Got my output | Receipt = " + std::to_string(mOut[i]));
                mOut.erase(mOut.begin() + i);
                return;
            }
        }

        lk.unlock();
        print(title + "Finish");
    }
}