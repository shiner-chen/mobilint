#include "demo/demo.h"

#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/feeder.h"
#include "demo/model.h"
#include "maccel/maccel.h"
#include "opencv2/opencv.hpp"

using mobilint::Accelerator;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::ModelConfig;
using mobilint::StatusCode;
using namespace std;

namespace {
void sleepForMS(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

std::string fpsToString(float fps) {
    char buf[20];
    sprintf(buf, "%8.2f", fps);
    return std::string(buf);
}

std::string secToString(int sec) {
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;

    char buf[20];
    sprintf(buf, "%02d:%02d:%02d", h, m, s);
    return std::string(buf);
}

std::string countToString(int count) {
    char buf[20];
    sprintf(buf, "%8d", count);
    return std::string(buf);
}

// ROI 사이즈 변화에 대해서 일정한 폰트를 유지할 수 있도록 한다.
// - 고정된 사이즈(w, h)에 대해 해당 폰트 사이즈와 굵기로 Benchmark 창을 만든다.
// - 필요한 만큼만 Benchmark 창을 Clip한다.
// - Clip한 창을 scale만큼 resize 후 frame에 띄운다.
void displayBenchmark(Item& item, bool is_fps_only = false) {
    float scale = 0.55;
    int w = 300;
    int h = 200;
    double font_scale = 1.0;
    int font_thickness = 1;

    cv::Mat board = cv::Mat::zeros(h, w, CV_8UC3);

    putText(board, "FPS", cv::Point(15, 40), cv::FONT_HERSHEY_DUPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness);
    putText(board, fpsToString(item.fps), cv::Point(112, 40), cv::FONT_HERSHEY_DUPLEX,
            font_scale, cv::Scalar(0, 255, 0), font_thickness);

    if (!is_fps_only) {
        putText(board, "Time", cv::Point(15, 80), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, secToString(item.time), cv::Point(110, 80),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0),
                font_thickness);

        putText(board, "Infer", cv::Point(15, 115), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, countToString(item.count), cv::Point(112, 115),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0),
                font_thickness);
    }

    int clip_w = 265;
    int clip_h;
    if (is_fps_only) {
        clip_h = 60;
    } else {
        clip_h = 94;
    }

    cv::Mat clip = board({0, 0, clip_w, clip_h});

    float resize_scale = (float)item.img.size().width / w * scale;
    // 이미지가 커지만 Debug 창도 같이 커진다.
    // 일정이상 비율이라면 더 이상 커지지 않게끔 한다.
    if (resize_scale > 0.9) {
        resize_scale = 0.9;
    }
    cv::resize(clip, clip, {0, 0}, resize_scale, resize_scale);

    int offset = (int)(item.img.size().width * 0.03);
    cv::Mat roi = item.img({{offset, offset}, clip.size()});
    cv::addWeighted(clip, 1, roi, 0.5, 0, roi);
}

void displayTime(cv::Mat& display, bool validate, float time = 0.0f) {
    float scale = 0.20;
    int w = 300;
    int h = 200;

    cv::Mat board = cv::Mat::zeros(h, w, CV_8UC3);
    if (validate) {
        double font_scale = 1.0;
        int font_thickness = 1;
        putText(board, "Time", cv::Point(15, 40), cv::FONT_HERSHEY_DUPLEX, font_scale,
                cv::Scalar(255, 255, 255), font_thickness);
        putText(board, secToString(time), cv::Point(110, 40), cv::FONT_HERSHEY_DUPLEX,
                font_scale, cv::Scalar(0, 255, 0), font_thickness);
    } else {
        board = cv::Scalar(255, 255, 255);
    }

    cv::Size size = display.size();
    int clip_w = 265;
    int clip_h = 60;
    cv::Mat clip = board({0, 0, clip_w, clip_h});
    float resize_scale = (float)size.width * scale / w;
    cv::resize(clip, clip, {0, 0}, resize_scale, resize_scale);

    int x = (int)(size.width * 0.25);
    int y = (int)(size.height * 0.3);
    cv::Mat roi = display({{x, y}, clip.size()});
    clip.copyTo(roi);

    if (!validate) {
        sleepForMS(50);
    }
}
}  // namespace

Demo::Demo()
    : mDisplayFPSMode(false),
      mDisplayTimeMode(false),
      mModeIndex(-1),
      // TODO: Thread 수를 어떻게 결정하지?
      // 일단은 최대 30개일테니 30으로 하자.
      mThreadPool(30) {}

void Demo::startWorker(int index) {
    const WorkerLayout& wl = mLayoutSetting.worker_layout[index];

    if (wl.model_index >= mModels.size() || wl.feeder_index >= mFeeders.size()) {
        return;
    }

    auto it = mModelFutureMap.find(index);
    if (it == mModelFutureMap.end()) {
        mSizeState[index]->open();
        auto future = mThreadPool.enqueue(
            Model::work, mModels[wl.model_index].get(), index, mSizeState[index].get(),
            &mMainQueueForWorker, &mFeeders[wl.feeder_index]->getMatBuffer());
        mModelFutureMap.emplace(index, std::move(future));
    }
}

void Demo::stopWorker(int index) {
    auto it = mModelFutureMap.find(index);
    if (it != mModelFutureMap.end()) {
        mSizeState[index]->close();
        if (it->second.valid()) {
            it->second.get();
        }
        mModelFutureMap.erase(it);
    }
}

void Demo::startFeeder(int index) {
    // 해당 조건으로 인해 feeder가 feeder_layout보다 많을 경우 size는 (0,0)이 되며,
    // size가 (0,0)일 경우 feeder는 display하지 않는다.
    cv::Size size;
    if (index < mLayoutSetting.feeder_layout.size()) {
        size = mLayoutSetting.feeder_layout[index].roi.size();
    }

    auto it = mFeederThreadMap.find(index);
    if (it == mFeederThreadMap.end()) {
        mFeeders[index]->start();
        mFeederThreadMap.emplace(
            index, std::thread(
                       [=] { mFeeders[index]->feed(index, mMainQueueForFeeder, size); }));
    }
}

void Demo::stopFeeder(int index) {
    auto it = mFeederThreadMap.find(index);
    if (it != mFeederThreadMap.end()) {
        mFeeders[index]->stop();
        it->second.join();
        mFeederThreadMap.erase(it);
    }
}

void Demo::startWorkerAll() {
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        startWorker(i);
    }
}

void Demo::stopWorkerAll() {
    for (int i = 0; i < mSizeState.size(); i++) {
        mSizeState[i]->close();
    }

    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        stopWorker(i);
    }
}

void Demo::startFeederAll() {
    for (int i = 0; i < mFeeders.size(); i++) {
        startFeeder(i);
    }
}

void Demo::stopFeederAll() {
    for (int i = 0; i < mFeeders.size(); i++) {
        stopFeeder(i);
    }
}

void Demo::startThreads() {
    mWorkerWatchdog = std::thread(&Demo::workerReceive, this);
    mFeederWatchdog = std::thread(&Demo::feederReceive, this);
}

void Demo::joinThreads() {
    mMainQueueForWorker.close();
    mMainQueueForFeeder.close();
    mWorkerWatchdog.join();
    mFeederWatchdog.join();
}

int Demo::getWorkerIndex(int x, int y) {
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        if (mLayoutSetting.worker_layout[i].roi.contains({x, y})) {
            return i;
        }
    }
    return -1;
}

void Demo::onMouseEvent(int event, int x, int y, int flags, void* ctx) {
    if (event != cv::EVENT_RBUTTONDOWN && event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    Demo* demo = (Demo*)ctx;
    int worker_index = demo->getWorkerIndex(x, y);
    if (worker_index == -1) {
        return;
    }

    switch (event) {
    case cv::EVENT_RBUTTONDOWN:
        demo->stopWorker(worker_index);
        break;
    case cv::EVENT_LBUTTONDOWN:
        demo->startWorker(worker_index);
        break;
    }
}

void Demo::feederReceive() {
    ItemQueue::StatusCode sc;
    Item item;
    while (true) {
        sc = mMainQueueForFeeder.pop(item);
        if (sc != ItemQueue::StatusCode::OK) {
            break;
        }

        if (mDisplayFPSMode) {
            displayBenchmark(item, true);
        }

        unique_lock<mutex> lock(mDisplayMutex);
        item.img.copyTo(mDisplay(mLayoutSetting.feeder_layout[item.index].roi));
    }
}

void Demo::workerReceive() {
    ItemQueue::StatusCode sc;
    Item item;
    while (true) {
        sc = mMainQueueForWorker.pop(item);
        if (sc != ItemQueue::StatusCode::OK) {
            break;
        }

        cv::Mat roi = mDisplay(mLayoutSetting.worker_layout[item.index].roi);

        // worker는 종료되는 시점에서 Mat()을 push한다.
        if (item.img.empty()) {
            unique_lock<mutex> lock(mDisplayMutex);
            roi = cv::Scalar(255, 255, 255);  // clear
            continue;
        }

        // 다른 사이즈의 img는 스킵한다.
        if (roi.size() != item.img.size()) {
            continue;
        }

        if (mDisplayFPSMode) {
            displayBenchmark(item);
        }
        unique_lock<mutex> lock(mDisplayMutex);
        item.img.copyTo(roi);
    }
}

void Demo::initWindow() {
    cv::Size window_size(1920, 1080);

    // Init Window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow(WINDOW_NAME, window_size / 2);
    cv::moveWindow(WINDOW_NAME, 0, 0);
    cv::setMouseCallback(WINDOW_NAME, onMouseEvent, this);

    mDisplay = cv::Mat(window_size, CV_8UC3, {255, 255, 255});

    // TODO: splash path 하드코딩;
    mSplashes.clear();
    for (string path : {"../rc/layout/splash_01.png", "../rc/layout/splash_02.png"}) {
        cv::Mat splash = cv::imread(path);
        cv::resize(splash, splash, cv::Size(1920, 1080));
        mSplashes.push_back(splash);
    }
}

void Demo::initLayout(std::string path) {
    mLayoutSetting = loadLayoutSettingYAML(path);

    // Clear
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mDisplay.setTo(cv::Scalar(255, 255, 255));
    }

    // Draw Banner
    for (const auto& il : mLayoutSetting.image_layout) {
        il.img.copyTo(mDisplay(il.roi));
    }

    // SizeState
    mSizeState.clear();
    mSizeState.resize(mLayoutSetting.worker_layout.size());
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        mSizeState[i] = make_unique<SizeState>();
        mSizeState[i]->update(mLayoutSetting.worker_layout[i].roi.size());
    }
}

void Demo::initModels(std::string path) {
    mModelSetting = loadModelSettingYAML(path);

    mModels.clear();
    mAccs.clear();
    mModels.resize(mModelSetting.size());
    for (int i = 0; i < mModelSetting.size(); i++) {
        int dev_no = mModelSetting[i].dev_no;
        auto it = mAccs.find(dev_no);
        if (it == mAccs.end()) {
            StatusCode sc;
            mAccs.emplace(dev_no, Accelerator::create(dev_no, sc));
        }
        mModels[i] = std::make_unique<Model>(mModelSetting[i], *mAccs[dev_no]);
    }
}

void Demo::initFeeders(std::string path) {
    mFeederSetting = loadFeederSettingYAML(path);

    mFeeders.resize(mFeederSetting.size());
    for (int i = 0; i < mFeederSetting.size(); i++) {
        mFeeders[i] = make_unique<Feeder>(mFeederSetting[i]);
    }
}

void Demo::display() {
    unique_lock<mutex> lock(mDisplayMutex);
    if (mDisplayTimeMode) {
        displayTime(mDisplay, true, mBenchmarker.getTimeSinceCreated());
    }
    cv::imshow(WINDOW_NAME, mDisplay);
}

void Demo::toggleDisplayFPSMode() { mDisplayFPSMode = !mDisplayFPSMode; }

void Demo::toggleDisplayTimeMode() {
    mDisplayTimeMode = !mDisplayTimeMode;
    if (!mDisplayTimeMode) {
        displayTime(mDisplay, false);
    }
}

void Demo::toggleScreenSize() {
    int cur = cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, !cur);
}

bool Demo::keyHandler(int key) {
    if (key == -1) {
        return true;
    }

    if (key >= 128) {  // Numpad 반환값은 128을 빼서 사용
        key -= 128;
    }

    key = tolower(key);

    if (key == 'd') {  // 'D'ebug
        toggleDisplayFPSMode();
    } else if (key == 't') {  // 'T'ime
        toggleDisplayTimeMode();
    } else if (key == 'm') {  // 'M'aximize Screen
        toggleScreenSize();
    } else if (key == 'c') {  // 'C'lear
        stopWorkerAll();
    } else if (key == 'f') {  // 'F'ill Grid
        startWorkerAll();
    } else if (key == 'q') {  // 'Q'uit
        stopWorkerAll();
        stopFeederAll();
        return false;
    } else if (key == '1' || key == '2' || key == '3') {
        setMode(key - '0');
    }

    return true;
}

void Demo::setMode(int mode_index) {
    // clang-format off
    switch (mode_index) {
        case 1: setMode1(); break;
        case 2: setMode2(); break;
        case 3: setMode3(); break;
    }
    // clang-format on
}

void Demo::setMode1() {
    if (mModeIndex == 1) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[0].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting.yaml");
    if (mModeIndex == -1 || mModeIndex == 3) {
        initModels("../rc/ModelSetting.yaml");
    }
    startWorkerAll();
    mModeIndex = 1;
    sleepForMS(500);
}

void Demo::setMode2() {
    if (mModeIndex == 2) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[0].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting2.yaml");
    if (mModeIndex == -1 || mModeIndex == 3) {
        initModels("../rc/ModelSetting.yaml");
    }
    startWorkerAll();
    mModeIndex = 2;
    sleepForMS(500);
}

void Demo::setMode3() {
    if (mModeIndex == 3) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[1].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting3.yaml");
    initModels("../rc/ModelSetting2.yaml");
    startWorkerAll();
    mModeIndex = 3;
    sleepForMS(500);
}

void Demo::run() {
    initWindow();
    initLayout("../rc/LayoutSetting.yaml");
    initModels("../rc/ModelSetting.yaml");
    initFeeders("../rc/FeederSetting.yaml");

    startFeederAll();

    startThreads();

    while (true) {
        display();
        if (!keyHandler(cv::waitKey(10))) {  // 1일 경우, 600 fps 이상이 나온다.
            break;
        }
    }

    joinThreads();

    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    Demo demo;
    demo.run();
    return 0;
}
