#include "demo/model.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "demo/post_ssd.h"
#include "demo/post_yolov5_face.h"
#include "demo/post_yolov8_pose.h"
#include "demo/post_yolov8_seg.h"
#include "opencv2/opencv.hpp"

Model::Model(const ModelSetting& model_setting, mobilint::Accelerator& acc) {
    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;
    mc.excludeAllCores();

    for (auto core_id : model_setting.core_id) {
        mc.include(core_id.cluster, core_id.core);
    }

    mModel = mobilint::Model::create(model_setting.mxq_path, mc, sc);
    mModel->launch(acc);

    // clang-format off
    switch (model_setting.model_type) {
    case ModelType::SSD         : initSSD();   break;
    case ModelType::STYLENET    : initStyle(); break;
    case ModelType::FACENET     : initFace();  break;
    case ModelType::POSE        : initPose();  break;
    case ModelType::SEGMENTATION: initSeg();   break;
    }
    // clang-format on
}

Model::~Model() { mModel->dispose(); }

void Model::initSSD() {
    mPost = std::make_unique<SSDPostProcessor>();
    mInference = &Model::inferenceSSD;
}

void Model::initStyle() { mInference = &Model::inferenceStyle; }

void Model::initFace() {
    float face_conf_thres = 0.15;
    float face_iou_thres = 0.35;
    int face_nl = 3;   // number of detection layers
    int face_nc = 1;   // number of classes
    int face_no = 16;  // number outputs per anchor
    int face_imh = 512;
    int face_imw = 640;
    mPost = std::make_unique<YOLOv5FacePostProcessor>(face_nl, face_nc, face_no, face_imh,
                                                      face_imw, face_conf_thres,
                                                      face_iou_thres, false);
    mInference = &Model::inferenceFace;
}

void Model::initPose() {
    float pose_conf_thres = 0.25;
    float pose_iou_thres = 0.65;
    int pose_nc = 1;  // number of classes
    int pose_imh = 512;
    int pose_imw = 640;
    mPost = std::make_unique<mobilint::post::YOLOv8PosePostProcessor>(
        pose_nc, pose_imh, pose_imw, pose_conf_thres, pose_iou_thres, false);
    mInference = &Model::inferencePose;
}

void Model::initSeg() {
    float seg_conf_thres = 0.20;
    float seg_iou_thres = 0.55;
    int seg_nc = 80;  // number of classes
    int seg_imh = 512;
    int seg_imw = 640;
    mPost = std::make_unique<mobilint::post::YOLOv8SegPostProcessor>(
        seg_nc, seg_imh, seg_imw, seg_conf_thres, seg_iou_thres, false);
    mInference = &Model::inferenceSeg;
}

void Model::work(Model* model, int worker_index, SizeState* size_state,
                 ItemQueue* item_queue, MatBuffer* feeder_buffer) {
    Benchmarker benchmarker;

    cv::Mat frame, result;
    cv::Size result_size;

    int64_t frame_index = 0;
    while (true) {
        // workerReceive 함수에서 Mat()를 받으면 worker가 죽은 것으로 간주하고 화면을
        // clear한다.
        auto ssc = size_state->checkUpdate(result_size);
        if (ssc != SizeState::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        auto msc = feeder_buffer->get(frame, frame_index);
        if (msc != MatBuffer::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        benchmarker.start();
#ifdef USE_SLEEP_DRIVER
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cv::resize(frame, result, result_size);
#else
        result = model->inference(frame, result_size);
#endif
        benchmarker.end();

        item_queue->push({worker_index, result, benchmarker.getFPS(),
                          benchmarker.getTimeSinceCreated(), benchmarker.getCount()});
    }
}

cv::Mat Model::inference(cv::Mat frame, cv::Size size) {
    return (this->*mInference)(frame, size);
}