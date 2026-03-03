#ifndef DEMO_INCLUDE_MODEL_H_
#define DEMO_INCLUDE_MODEL_H_

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "qbruntime/model.h"
#include "opencv2/opencv.hpp"
#include "post.h"

class Model {
public:
    Model() = delete;
    Model(const ModelSetting& model_setting, mobilint::Accelerator& acc);
    ~Model();

    static void work(Model* model, int worker_index, SizeState* size_state,
                     ItemQueue* item_queue, MatBuffer* feeder_buffer);

    cv::Mat inference(cv::Mat frame, cv::Size size);

private:
    cv::Mat (Model::*mInference)(cv::Mat, cv::Size);

    std::unique_ptr<mobilint::Model> mModel;
    std::unique_ptr<PostProcessor> mPost;

    void initSSD();
    void initStyle();
    void initFace();
    void initPose();
    void initSeg();

    cv::Mat inferenceSSD(cv::Mat frame, cv::Size size);
    cv::Mat inferenceStyle(cv::Mat frame, cv::Size size);
    cv::Mat inferenceFace(cv::Mat frame, cv::Size size);
    cv::Mat inferencePose(cv::Mat frame, cv::Size size);
    cv::Mat inferenceSeg(cv::Mat frame, cv::Size size);
};
#endif
