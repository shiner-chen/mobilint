#include "demo/model.h"
#include "opencv2/opencv.hpp"

cv::Mat Model::inferenceSSD(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = ((float)resized_frame.data[i] - 127.5) / 127.5;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef USE_ARIES2
    result = {std::move(result[9]), std::move(result[8]),  std::move(result[7]),
              std::move(result[6]), std::move(result[5]),  std::move(result[4]),
              std::move(result[3]), std::move(result[2]),  std::move(result[1]),
              std::move(result[0]), std::move(result[11]), std::move(result[10])};
#endif

    std::vector<float> boxes, classes, scores;
    uint64_t ticket = mPost->enqueue(result, boxes, classes, scores);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    cv::Point pt1, pt2;
    for (int i = 0; i < scores.size(); i++) {
        if (classes[i] != 1) {
            continue;
        }
        pt1.x = boxes[i * 4 + 0] * size.width;
        pt1.y = boxes[i * 4 + 1] * size.height;
        pt2.x = boxes[i * 4 + 2] * size.width;
        pt2.y = boxes[i * 4 + 3] * size.height;

        cv::rectangle(result_frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    return result_frame;
}

cv::Mat Model::inferenceStyle(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;
    int wi = mModel->getInputBufferInfo()[0].original_width;
    int hi = mModel->getInputBufferInfo()[0].original_height;
    int ci = mModel->getInputBufferInfo()[0].original_channel;

    int wo = mModel->getOutputBufferInfo()[0].original_width;
    int ho = mModel->getOutputBufferInfo()[0].original_height;
    int co = mModel->getOutputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(wi, hi));

    auto input_img = std::make_unique<float[]>(wi * hi * ci);
    for (int i = 0; i < wi * hi; i++) {
        // BGR -> RGB
        input_img.get()[i * 3 + 0] = (float)resized_frame.data[i * 3 + 2] / 255.0f;
        input_img.get()[i * 3 + 1] = (float)resized_frame.data[i * 3 + 1] / 255.0f;
        input_img.get()[i * 3 + 2] = (float)resized_frame.data[i * 3 + 0] / 255.0f;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    for (int i = 0; i < wo * ho; i++) {
        // RGB -> BGR
        resized_frame.data[i * 3 + 0] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 2] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 1] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 1] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 2] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 0] * 255.0f, 255.0f));
    }

    // 추가된 Style Change 모델은 가장자리에 이물이 남아 이를 crop하여 사용하기로 한다.
    int crop_x = 35;
    int crop_y = 20;
    int crop_w = wo - crop_x * 2;
    int crop_h = ho - crop_y * 2;
    cv::Mat cropped_frame = resized_frame(cv::Rect{crop_x, crop_y, crop_w, crop_h});

    cv::Mat result_frame;
    cv::resize(cropped_frame, result_frame, size);

    return result_frame;
}

cv::Mat Model::inferenceFace(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512(480 + 2 * 16)
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    int y_pad = 16;
    int h_pad = h - y_pad * 2;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h_pad));

    // 480 이미지 위 아래에 16만큼 zero padding을 한다.
    // 512만한 pad 이미지에 resized_frame을 붙여넣기 한다.
    cv::Mat padded_resized_frame = cv::Mat::zeros(h, w, CV_8UC3);
    resized_frame.copyTo(padded_resized_frame({0, y_pad, w, h_pad}));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = (float)padded_resized_frame.data[i] / 255;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> landmarks;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, landmarks);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}

cv::Mat Model::inferencePose(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h; i++) {
        int idx = i * 3;
        // BGR -> RGB 배열 변환
        input_img.get()[idx + 0] = ((float)resized_frame.data[idx + 2] / 255);
        input_img.get()[idx + 1] = ((float)resized_frame.data[idx + 1] / 255);
        input_img.get()[idx + 2] = ((float)resized_frame.data[idx + 0] / 255);
    }

    auto result =
        mModel->infer({input_img.get()}, sc);  // vector<vector<float>> {0, 1, 3, 2}
    if (!sc) {
        std::cout << "infer failed" << std::endl;
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef USE_ARIES2
    result = {
        std::move(result[1]), std::move(result[3]), std::move(result[5]),
        std::move(result[0]), std::move(result[2]), std::move(result[4]),
    };
#endif

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> keypoints;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, keypoints);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}

cv::Mat Model::inferenceSeg(cv::Mat frame, cv::Size size) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;    // 640
    int h = mModel->getInputBufferInfo()[0].original_height;   // 512(480 + 2 * 16)
    int c = mModel->getInputBufferInfo()[0].original_channel;  // 3

    int y_pad = 16;
    int h_pad = h - y_pad * 2;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h_pad));

    // 480 이미지 위 아래에 16만큼 zero padding을 한다.
    // 512만한 pad 이미지에 resized_frame을 붙여넣기 한다.
    cv::Mat padded_resized_frame = cv::Mat::zeros(h, w, CV_8UC3);
    resized_frame.copyTo(padded_resized_frame({0, y_pad, w, h_pad}));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = (float)padded_resized_frame.data[i] / 255;
    }

    auto result = mModel->infer({input_img.get()}, sc);
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

#ifdef USE_ARIES2
    result = {
        std::move(result[2]), std::move(result[1]), std::move(result[4]),
        std::move(result[6]), std::move(result[0]), std::move(result[3]),
        std::move(result[5]),
    };
#endif

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;
    uint64_t ticket =
        mPost->enqueue(resized_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(resized_frame, result_frame, size);

    return result_frame;
}
