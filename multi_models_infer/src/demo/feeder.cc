#include "demo/feeder.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

namespace {
std::string getYouTube(const std::string& youtube_url) {
#ifdef _MSC_VER
    // TODO(kibum): Need to implement.
    std::cerr << "Youtube input is not implemented for MSVC.\n";
    return "";
#else
    char buf[128];
    std::string URL;
    std::string cmd = "yt-dlp -f \"best[height<=720][width<=1280]\" -g " + youtube_url;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return URL;
    }

    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        URL += buf;
    }
    pclose(pipe);

    if (!URL.empty()) {
        URL.erase(URL.find('\n'));
    }
    return URL;
#endif
}
}  // namespace

Feeder::Feeder(const FeederSetting& feeder_setting) : mFeederSetting(feeder_setting) {
    for (int i = 0; i < mFeederSetting.src_path.size(); i++) {
        cv::VideoCapture cap;
        switch (mFeederSetting.feeder_type) {
        case FeederType::CAMERA: {
            cap.open(stoi(mFeederSetting.src_path[i]), cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
            cap.set(cv::CAP_PROP_FPS, 30);
            mDelayOn = false;
            break;
        }
        case FeederType::VIDEO: {
            cap.open(mFeederSetting.src_path[i]);
            mDelayOn = true;
            break;
        }
        case FeederType::IPCAMERA: {
            cap.open(mFeederSetting.src_path[i]);
            mDelayOn = false;
            break;
        }
        case FeederType::YOUTUBE: {
            cap.open(getYouTube(mFeederSetting.src_path[i]));
            mDelayOn = true;
            break;
        }
        }
        mCap.push_back(cap);
    }
}

void Feeder::feed(int index, ItemQueue& item_queue, cv::Size roi_size) {
    mFeederBuffer.open();
    while (mIsFeederRunning) {
        for (int i = 0; i < mCap.size(); i++) {
            if (mCap[i].isOpened()) {
                feedInternal(index, item_queue, mCap[i], roi_size, mDelayOn);
                mCap[i].set(cv::CAP_PROP_POS_FRAMES, 0);
            } else {
                feedInternalDummy(index, item_queue, roi_size);
            }
        }
    }
    mFeederBuffer.close();
}

// TODO: 일단 인터넷 끊긴 후 회생 로직을 구현할 방법이 도저히 없다.
// void Feeder::feed(int index, ItemQueue& item_queue, cv::Size roi_size) {
//     int idx = 0;

//     int sleep_time = 1000;  // 1sec
//     while (mIsFeederRunning) {
//         cv::VideoCapture cap;
//         bool delay_on;
//         switch (mFeederSetting.feeder_type) {
//         case FeederType::CAMERA: {
//             cap.open(stoi(mFeederSetting.src_path[0]), cv::CAP_V4L2);
//             cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
//             cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
//             cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
//             cap.set(cv::CAP_PROP_FPS, 30);
//             delay_on = false;
//             break;
//         }
//         case FeederType::VIDEO: {
//             cap.open(mFeederSetting.src_path[idx++ % mFeederSetting.src_path.size()]);
//             delay_on = true;
//             break;
//         }
//         case FeederType::IPCAMERA: {
//             cap.open(mFeederSetting.src_path[0]);
//             delay_on = false;
//             break;
//         }
//         case FeederType::YOUTUBE: {
//             cap.open(getYouTube(
//                 mFeederSetting.src_path[idx++ % mFeederSetting.src_path.size()]));
//             delay_on = true;
//             break;
//         }
//         default: {
//             break;
//         }
//         }
//         if (cap.isOpened()) {
//             feedInternal(index, item_queue, cap, roi_size, delay_on);
//             sleep_time = 1000;
//         } else {
//             // 전원문제로 인터넷이 끊길 경우, CCTV Feeder가 죽어서 프로그램의 재실행이
//             // 필요합니다. 이 경우 Demo 수행시간이 초기화되는데, 재실행 없이 Feeder를
//             // 되살리기 위한 로직입니다. VideoCapture 객체의 open을 여러번 호출시
//             멈추게
//             // 되는데, 호출 주기를 줄이고자 workaround로 sleep을 하게됩니다.
//             std::cerr << "Source Open Error." << std::endl;
//             std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
//             sleep_time *= 2;
//             if (sleep_time > 600'000) {
//                 sleep_time = 600'000;  // maximum 10min
//             }
//         }
//         cap.release();
//     }

//     mFeederBuffer.close();
// }

void Feeder::feedInternal(int index, ItemQueue& item_queue, cv::VideoCapture& cap,
                          cv::Size roi_size, bool delay_on) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push({index, resized_frame, benchmarker.getFPS(), 0.0, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        if (delay_on) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        benchmarker.end();
    }
}

void Feeder::feedInternalDummy(int index, ItemQueue& item_queue, cv::Size roi_size) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190), cv::FONT_HERSHEY_DUPLEX,
                    1.5, cv::Scalar(0, 255, 0), 2);
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push({index, resized_frame, benchmarker.getFPS(), 0.0, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        benchmarker.end();
    }
}