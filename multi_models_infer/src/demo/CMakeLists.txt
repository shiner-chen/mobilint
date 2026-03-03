# default
set(VENDOR "mobilint" CACHE STRING "")
set(PRODUCT "aries2-v4" CACHE STRING "")
set(DRIVER_TYPE "aries2" CACHE STRING "")

string(TOUPPER ${DRIVER_TYPE} UPPERCASE_DRIVER_TYPE)

include(MobilintExtLib)
if(NOT DEFINED MACCEL_PATH)
    mobilint_extlib_declare(maccel GIT_TAG master)
    mobilint_extlib_make_available(maccel)
endif()
mobilint_extlib_declare(yaml-cpp)
mobilint_extlib_make_available(yaml-cpp)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(DEMO_SOURCES
    demo.cc
    feeder.cc

    model.cc
    model_inference.cc

    post_yolov5.cc
    post_yolov5_face.cc
    
    post_ssd.cc
    
    post_yolov8.cc
    post_yolov8_pose.cc
    post_yolov8_seg.cc

    yaml_util.cc
)
set(DEMO_EXTERNAL_LIBRARIES
    maccel
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    yaml-cpp
)

add_executable(demo ${DEMO_SOURCES})
if(DEFINED MACCEL_PATH)
    target_link_directories(demo PRIVATE "${PROJECT_BINARY_DIR}/${MACCEL_PATH}/lib")
    target_include_directories(demo PRIVATE "${PROJECT_BINARY_DIR}/${MACCEL_PATH}/include")
endif()
target_link_libraries(demo PRIVATE ${DEMO_EXTERNAL_LIBRARIES})
target_include_directories(demo PRIVATE "${PROJECT_SOURCE_DIR}/src")
# NOTE: Post-process codes from algorithm team use c++17 features.
target_compile_features(demo PRIVATE cxx_std_17)

if(PRODUCT STREQUAL "aries2-v4")
    target_compile_definitions(demo PRIVATE USE_ARIES2)
endif()

if(UPPERCASE_DRIVER_TYPE STREQUAL "SLEEP")
    target_compile_definitions(demo PRIVATE USE_SLEEP_DRIVER)
endif()
