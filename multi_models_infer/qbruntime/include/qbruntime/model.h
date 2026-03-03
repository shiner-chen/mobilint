// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.
/**
 * \file
 */

#ifndef QBRUNTIME_MODEL_H_
#define QBRUNTIME_MODEL_H_

#include <cstdint>
#ifndef _MSC_VER
#include <experimental/propagate_const>
#endif
#include <memory>
#include <string>
#include <vector>

#include "qbruntime/export.h"
#include "qbruntime/future.h"
#include "qbruntime/model_variant_handle.h"
#include "qbruntime/ndarray.h"
#include "qbruntime/status_code.h"
#include "qbruntime/type.h"

namespace mobilint {

/**
 * \addtogroup CPPAPI
 * @{
 */

class Accelerator;
class ModelImpl;

/**
 * @brief Represents an AI model loaded from an MXQ file.
 *
 * This class loads an AI model from an MXQ file and provides functions to launch it
 * on the NPU and perform inference.
 */
class QBRUNTIME_EXPORT Model {
public:
    /**
     * @brief Creates a Model object from the specified MXQ model file.
     *
     * Parses the MXQ file and constructs a Model object. The model is initialized in
     * single-core mode with all NPU local cores included.
     *
     * @note The created Model object must be launched before performing inference.
     *       See Model::launch for more details.
     *
     * @param[in] mxq_path The path to the MXQ model file.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the model was successfully created or if an error occurred.
     * @return A unique pointer to the created Model object.
     */
    static std::unique_ptr<Model> create(const std::string& mxq_path, StatusCode& sc);

    /**
     * @brief Creates a Model object from the specified MXQ model file and configuration.
     *
     * Parses the MXQ file and constructs a Model object using the provided configuration,
     * initializing the model with the given settings.
     *
     * @note The created Model object must be launched before performing inference.
     *       See Model::launch for more details.
     *
     * @param[in] mxq_path The path to the MXQ model file.
     * @param[in] config The configuration settings to initialize the Model.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the model was successfully created or if an error occurred.
     * @return A unique pointer to the created Model object.
     */
    static std::unique_ptr<Model> create(const std::string& mxq_path,
                                         const ModelConfig& config, StatusCode& sc);

    Model(const Model& other) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) noexcept;
    ~Model();

    /**
     * @brief Launches the model on the specified Accelerator, which represents
     * the actual NPU.
     *
     * @param[in] acc The accelerator on which to launch the model.
     * @return A status code indicating whether the model was successfully launched
     *         or if an error occurred.
     */
    StatusCode launch(Accelerator& acc);

    /**
     * @brief Disposes of the model loaded onto the NPU.
     *
     * Releases any resources associated with the model on the NPU.
     *
     * @return A status code indicating whether the disposal was successful
     *         or if an error occurred.
     */
    StatusCode dispose();

    /**
     * @brief Retrieves the core mode of the model.
     *
     * @return The CoreMode of the model.
     */
    CoreMode getCoreMode() const;

    /**
     * @brief Checks if the NPU core specified by CoreId is the target of the model.
     *        In other words, whether the model is configured to use the given NPU core.
     *
     * @param[in] core_id The CoreId to check.
     * @return True if the model is configured to use the specified CoreId, false
     * otherwise.
     */
    bool isTarget(CoreId core_id) const;

    /**
     * @brief Returns the NPU cores the model is configured to use.
     *
     * @return A vector of CoreIds representing the target NPU cores.
     */
    std::vector<CoreId> getTargetCores() const;

    /**
     * @name NHWC float-to-float inference
     *
     * Performs inference with input and output elements of type float
     * in NHWC (batch N, height H, width W, channels C) or HWC format.
     *
     * Two input-output type pairs are supported:
     *
     * 1. `std::vector<NDArray<float>>` for both input and output
     *    - Recommended approach, as `NDArray` allows the qbruntime
     *      to avoid unnecessary data copies internally.
     *
     * 2. `std::vector<float*>` for input and `std::vector<std::vector<float>>` for output
     *    - Provided for user convenience, but results in unavoidable extra
     *      copies within the qbruntime.
     */
    /**@{*/

    /**
     * @brief Performs inference.
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NHWC or HWC
     *                  format.
     * @param[out] output A reference to a vector of `NDArray<float>` that will store the
     *                    inference results.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode infer(const std::vector<NDArray<float>>& input,
                     std::vector<NDArray<float>>& output);

    /**
     * @brief This overload differs from the above function in that it directly returns
     * the inference results instead of modifying an output parameter.
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NHWC or HWC
     *                  format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of `NDArray<float>` containing the inference results.
     */
    std::vector<NDArray<float>> infer(const std::vector<NDArray<float>>& input,
                                      StatusCode& sc);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in HWC format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike the above overload, this function returns the inference results directly
     * instead of modifying an output parameter.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in HWC format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          StatusCode& sc);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike other overloads, this version allows explicitly specifying the shape of each
     * input data, which can be in NHWC or HWC format.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NHWC or HWC format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @param[in] shape A vector of vectors, where each inner vector specifies the
     *                  shape of the corresponding input data.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output,
                     const std::vector<std::vector<int64_t>>& shape);
    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike the above overload, this function returns the inference results directly
     * instead of modifying an output parameter.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NHWC or HWC format.
     * @param[in] shape A vector of vectors, where each inner vector specifies the
     *                  shape of the corresponding input data.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          const std::vector<std::vector<int64_t>>& shape,
                                          StatusCode& sc);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of NDArrays, where each NDArray represents input data
     *                  in NHWC or HWC format.
     * @param[out] output A reference to a vector of NDArrays that will store the
     *                    inference results.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode infer(const std::vector<NDArray<float>>& input,
                     std::vector<NDArray<float>>& output, uint32_t cache_size);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * Unlike the above overload, this function returns the inference results
     * directly instead of modifying an output parameter.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of NDArrays, where each NDArray represents input data
     *                  in NHWC or HWC format.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of NDArrays containing the inference results.
     */
    std::vector<NDArray<float>> infer(const std::vector<NDArray<float>>& input,
                                      uint32_t cache_size, StatusCode& sc);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NHWC or HWC format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @param[in] shape A vector of vectors, where each inner vector specifies the shape
     *                  of the corresponding input data.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output,
                     const std::vector<std::vector<int64_t>>& shape, uint32_t cache_size);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * Unlike the above overload, this function returns the inference results
     * directly instead of modifying an output parameter.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NHWC or HWC format.
     * @param[in] shape A vector of vectors, where each inner vector specifies the shape
     *                  of the corresponding input data.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          const std::vector<std::vector<int64_t>>& shape,
                                          uint32_t cache_size, StatusCode& sc);

    /**@}*/

    /**
     * @name NCHW float-to-float inference
     *
     * Performs inference with input and output elements of type float
     * in NCHW (batch N, channels C, height H, width W) or CHW format.
     *
     * Two input-output type pairs are supported:
     *
     * 1. `std::vector<NDArray<float>>` for both input and output
     *    - Recommended approach, as `NDArray` allows the qbruntime
     *      to avoid unnecessary data copies internally.
     *
     * 2. `std::vector<float*>` for input and `std::vector<std::vector<float>>` for output
     *    - Provided for user convenience, but results in unavoidable extra
     *      copies within the qbruntime.
     *
     * @note CHW is not the recommended format, as the NPU natively operates on
     *       HWC-ordered data. When input is provided in CHW format, it will be
     *       transposed internally, introducing additional overhead.
     *
     * @note If your data is in HWC format, use `Model::infer` instead of
     *       `Model::inferCHW`, as it avoids unnecessary format conversion.
     */
    /**@{*/

    /**
     * @brief Performs inference
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NCHW or CHW
     *                  format.
     * @param[out] output A reference to a vector of `NDArray<float>` that will store the
     *                    inference results.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode inferCHW(const std::vector<NDArray<float>>& input,
                        std::vector<NDArray<float>>& output);

    /**
     * @brief This overload differs from the above function in that it directly returns
     * the inference results instead of modifying an output parameter.
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NCHW or CHW
     *                  format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of `NDArray<float>` containing the inference results.
     */
    std::vector<NDArray<float>> inferCHW(const std::vector<NDArray<float>>& input,
                                         StatusCode& sc);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in CHW format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode inferCHW(const std::vector<float*>& input,
                        std::vector<std::vector<float>>& output);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike the above overload, this function returns the inference results directly
     * instead of modifying an output parameter.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in CHW format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> inferCHW(const std::vector<float*>& input,
                                             StatusCode& sc);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike other overloads, this version allows explicitly specifying the shape of each
     * input data, which can be in NCHW or CHW format.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NCHW or CHW format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @param[in] shape A vector of vectors, where each inner vector specifies the
     *                  shape of the corresponding input data.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode inferCHW(const std::vector<float*>& input,
                        std::vector<std::vector<float>>& output,
                        const std::vector<std::vector<int64_t>>& shape);

    /**
     * @brief This overload is provided for convenience but may result in additional data
     * copies within the qbruntime.
     *
     * Unlike the above overload, this function returns the inference results directly
     * instead of modifying an output parameter.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NCHW or CHW format.
     * @param[in] shape A vector of vectors, where each inner vector specifies the
     *                  shape of the corresponding input data.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> inferCHW(
        const std::vector<float*>& input, const std::vector<std::vector<int64_t>>& shape,
        StatusCode& sc);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of NDArrays, where each NDArray represents input data
     *                  in NCHW or CHW format.
     * @param[out] output A reference to a vector of NDArrays that will store the
     *                    inference results.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode inferCHW(const std::vector<NDArray<float>>& input,
                        std::vector<NDArray<float>>& output, uint32_t cache_size);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * Unlike the above overload, this function returns the inference results
     * directly instead of modifying an output parameter.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of NDArrays, where each NDArray represents input data
     *                  in NCHW or CHW format.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of NDArrays containing the inference results.
     */
    std::vector<NDArray<float>> inferCHW(const std::vector<NDArray<float>>& input,
                                         uint32_t cache_size, StatusCode& sc);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NCHW or CHW format.
     * @param[out] output A reference to a vector of float vectors that will store the
     *                    inference results.
     * @param[in] shape A vector of vectors, where each inner vector specifies the shape
     *                  of the corresponding input data.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @return A status code indicating whether the inference operation completed
     *         successfully or encountered an error.
     */
    StatusCode inferCHW(const std::vector<float*>& input,
                        std::vector<std::vector<float>>& output,
                        const std::vector<std::vector<int64_t>>& shape,
                        uint32_t cache_size);

    /**
     * @brief This overload supports inference with KV cache.
     *
     * Unlike the above overload, this function returns the inference results
     * directly instead of modifying an output parameter.
     *
     * @note This function is relevant for LLM models that use KV cache.
     *
     * @param[in] input A vector of float pointers, where each pointer represents input
     *                  data in NCHW or CHW format.
     * @param[in] shape A vector of vectors, where each inner vector specifies the shape
     *                  of the corresponding input data.
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the inference operation was successful or encountered an
     *                error.
     * @return A vector of float vectors containing the inference results.
     */
    std::vector<std::vector<float>> inferCHW(
        const std::vector<float*>& input, const std::vector<std::vector<int64_t>>& shape,
        uint32_t cache_size, StatusCode& sc);

    /**@}*/

    /**
     * @name NHWC uint8_t-to-float inference
     *
     * Performs inference with input and output elements of type `uint8_t`
     * in NHWC (batch N, height H, width W, channels C) or HWC format.
     *
     */
    /**@{*/

    StatusCode infer(const std::vector<NDArray<uint8_t>>& input,
                     std::vector<NDArray<float>>& output);
    std::vector<NDArray<float>> infer(const std::vector<NDArray<uint8_t>>& input,
                                      StatusCode& sc);
    StatusCode infer(const std::vector<uint8_t*>& input,
                     std::vector<std::vector<float>>& output);
    std::vector<std::vector<float>> infer(const std::vector<uint8_t*>& input,
                                          StatusCode& sc);
    StatusCode infer(const std::vector<uint8_t*>& input,
                     std::vector<std::vector<float>>& output,
                     const std::vector<std::vector<int64_t>>& shape);
    std::vector<std::vector<float>> infer(const std::vector<uint8_t*>& input,
                                          const std::vector<std::vector<int64_t>>& shape,
                                          StatusCode& sc);

    StatusCode infer(const std::vector<NDArray<uint8_t>>& input,
                     std::vector<NDArray<float>>& output, uint32_t cache_size);
    std::vector<NDArray<float>> infer(const std::vector<NDArray<uint8_t>>& input,
                                      uint32_t cache_size, StatusCode& sc);
    StatusCode infer(const std::vector<uint8_t*>& input,
                     std::vector<std::vector<float>>& output,
                     const std::vector<std::vector<int64_t>>& shape, uint32_t cache_size);
    std::vector<std::vector<float>> infer(const std::vector<uint8_t*>& input,
                                          const std::vector<std::vector<int64_t>>& shape,
                                          uint32_t cache_size, StatusCode& sc);

    /**@}*/

    /**
     * @name NCHW uint8_t-to-float inference
     *
     * Performs inference with input and output elements of type `uint8_t`
     * in NCHW (batch N, channels C, height H, width W) or CHW format.
     *
     */
    /**@{*/
    StatusCode inferCHW(const std::vector<NDArray<uint8_t>>& input,
                        std::vector<NDArray<float>>& output);
    std::vector<NDArray<float>> inferCHW(const std::vector<NDArray<uint8_t>>& input,
                                         StatusCode& sc);
    StatusCode inferCHW(const std::vector<uint8_t*>& input,
                        std::vector<std::vector<float>>& output);
    std::vector<std::vector<float>> inferCHW(const std::vector<uint8_t*>& input,
                                             StatusCode& sc);
    StatusCode inferCHW(const std::vector<uint8_t*>& input,
                        std::vector<std::vector<float>>& output,
                        const std::vector<std::vector<int64_t>>& shape);
    std::vector<std::vector<float>> inferCHW(
        const std::vector<uint8_t*>& input,
        const std::vector<std::vector<int64_t>>& shape, StatusCode& sc);

    StatusCode inferCHW(const std::vector<NDArray<uint8_t>>& input,
                        std::vector<NDArray<float>>& output, uint32_t cache_size);
    std::vector<NDArray<float>> inferCHW(const std::vector<NDArray<uint8_t>>& input,
                                         uint32_t cache_size, StatusCode& sc);
    StatusCode inferCHW(const std::vector<uint8_t*>& input,
                        std::vector<std::vector<float>>& output,
                        const std::vector<std::vector<int64_t>>& shape,
                        uint32_t cache_size);
    std::vector<std::vector<float>> inferCHW(
        const std::vector<uint8_t*>& input,
        const std::vector<std::vector<int64_t>>& shape, uint32_t cache_size,
        StatusCode& sc);
    /**@}*/

    /**
     * @name NHWC int8_t-to-int8_t inference
     *
     * Performs inference with input and output elements of type `int8_t`
     * in NHWC (batch N, height H, width W, channels C) or HWC format.
     *
     * Using these inference APIs requires manual scaling (quantization)
     * of float values to `int8_t` for input and `int8_t` to float for output.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/

    StatusCode infer(const std::vector<NDArray<int8_t>>& input,
                     std::vector<NDArray<int8_t>>& output);
    std::vector<NDArray<int8_t>> infer(const std::vector<NDArray<int8_t>>& input,
                                       StatusCode& sc);
    StatusCode infer(const std::vector<int8_t*>& input,
                     std::vector<std::vector<int8_t>>& output);
    std::vector<std::vector<int8_t>> infer(const std::vector<int8_t*>& input,
                                           StatusCode& sc);
    StatusCode infer(const std::vector<int8_t*>& input,
                     std::vector<std::vector<int8_t>>& output,
                     const std::vector<std::vector<int64_t>>& shape);
    std::vector<std::vector<int8_t>> infer(const std::vector<int8_t*>& input,
                                           const std::vector<std::vector<int64_t>>& shape,
                                           StatusCode& sc);

    StatusCode infer(const std::vector<NDArray<int8_t>>& input,
                     std::vector<NDArray<int8_t>>& output, uint32_t cache_size);
    std::vector<NDArray<int8_t>> infer(const std::vector<NDArray<int8_t>>& input,
                                       uint32_t cache_size, StatusCode& sc);
    StatusCode infer(const std::vector<int8_t*>& input,
                     std::vector<std::vector<int8_t>>& output,
                     const std::vector<std::vector<int64_t>>& shape, uint32_t cache_size);
    std::vector<std::vector<int8_t>> infer(const std::vector<int8_t*>& input,
                                           const std::vector<std::vector<int64_t>>& shape,
                                           uint32_t cache_size, StatusCode& sc);

    /**@}*/

    /**
     * @name NCHW int8_t-to-int8_t inference
     *
     * Performs inference with input and output elements of type `int8_t`
     * in NCHW (batch N, channels C, height H, width W) or CHW format.
     *
     * Using these inference APIs requires manual scaling (quantization)
     * of float values to `int8_t` for input and `int8_t` to float for output.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/
    StatusCode inferCHW(const std::vector<NDArray<int8_t>>& input,
                        std::vector<NDArray<int8_t>>& output);
    std::vector<NDArray<int8_t>> inferCHW(const std::vector<NDArray<int8_t>>& input,
                                          StatusCode& sc);
    StatusCode inferCHW(const std::vector<int8_t*>& input,
                        std::vector<std::vector<int8_t>>& output);
    std::vector<std::vector<int8_t>> inferCHW(const std::vector<int8_t*>& input,
                                              StatusCode& sc);
    StatusCode inferCHW(const std::vector<int8_t*>& input,
                        std::vector<std::vector<int8_t>>& output,
                        const std::vector<std::vector<int64_t>>& shape);
    std::vector<std::vector<int8_t>> inferCHW(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        StatusCode& sc);

    StatusCode inferCHW(const std::vector<NDArray<int8_t>>& input,
                        std::vector<NDArray<int8_t>>& output, uint32_t cache_size);
    std::vector<NDArray<int8_t>> inferCHW(const std::vector<NDArray<int8_t>>& input,
                                          uint32_t cache_size, StatusCode& sc);
    StatusCode inferCHW(const std::vector<int8_t*>& input,
                        std::vector<std::vector<int8_t>>& output,
                        const std::vector<std::vector<int64_t>>& shape,
                        uint32_t cache_size);
    std::vector<std::vector<int8_t>> inferCHW(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        uint32_t cache_size, StatusCode& sc);
    /**@}*/

    /**
     * @name NHWC int8_t-to-float inference
     *
     * Performs inference with input and output elements of type `int8_t`
     * in NHWC (batch N, height H, width W, channels C) or HWC format.
     *
     * Using these inference APIs requires manual scaling (quantization)
     * of float values to `int8_t` for input.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/
    std::vector<NDArray<float>> inferToFloat(const std::vector<NDArray<int8_t>>& input,
                                             StatusCode& sc);
    std::vector<std::vector<float>> inferToFloat(const std::vector<int8_t*>& input,
                                                 StatusCode& sc);
    std::vector<std::vector<float>> inferToFloat(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        StatusCode& sc);

    std::vector<NDArray<float>> inferToFloat(const std::vector<NDArray<int8_t>>& input,
                                             uint32_t cache_size, StatusCode& sc);
    std::vector<std::vector<float>> inferToFloat(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        uint32_t cache_size, StatusCode& sc);
    /**@}*/

    /**
     * @name NCHW int8_t-to-float inference
     *
     * Performs inference with input and output elements of type `int8_t`
     * in NCHW (batch N, channels C, height H, width W) or CHW format.
     *
     * Using these inference APIs requires manual scaling (quantization)
     * of float values to `int8_t` for input.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/
    std::vector<NDArray<float>> inferCHWToFloat(const std::vector<NDArray<int8_t>>& input,
                                                StatusCode& sc);
    std::vector<std::vector<float>> inferCHWToFloat(const std::vector<int8_t*>& input,
                                                    StatusCode& sc);
    std::vector<std::vector<float>> inferCHWToFloat(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        StatusCode& sc);

    std::vector<NDArray<float>> inferCHWToFloat(const std::vector<NDArray<int8_t>>& input,
                                                uint32_t cache_size, StatusCode& sc);
    std::vector<std::vector<float>> inferCHWToFloat(
        const std::vector<int8_t*>& input, const std::vector<std::vector<int64_t>>& shape,
        uint32_t cache_size, StatusCode& sc);
    /**@}*/

    /**
     * @name NHWC Buffer-to-Buffer inference
     *
     * Performs inference using input and output elements in the NPU’s internal data type.
     * The inference operates on buffers allocated via the following APIs:
     *
     * - `Model::acquireInputBuffer`
     * - `Model::acquireOutputBuffer`
     * - `Model::acquireInputBuffers`
     * - `Model::acquireOutputBuffers`
     * - `ModelVariantHandle::acquireInputBuffer`
     * - `ModelVariantHandle::acquireOutputBuffer`
     * - `ModelVariantHandle::acquireInputBuffers`
     * - `ModelVariantHandle::acquireOutputBuffers`
     *
     * Additionally, `Model::repositionInputs`, `Model::repositionOutputs`,
     * `ModelVariantHandle::repositionInputs` and `ModelVariantHandle::repositionOutputs`
     * must be used properly.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/
    StatusCode inferBuffer(const std::vector<Buffer>& input, std::vector<Buffer>& output,
                           const std::vector<std::vector<int64_t>>& shape = {},
                           uint32_t cache_size = 0);
    StatusCode inferBuffer(const std::vector<std::vector<Buffer>>& input,
                           std::vector<std::vector<Buffer>>& output,
                           const std::vector<std::vector<int64_t>>& shape = {},
                           uint32_t cache_size = 0);
    /**@}*/

    /**
     * @name NHWC Buffer-to-float inference
     *
     * Performs inference using input and output elements in the NPU’s internal data type.
     * The inference operates on buffers allocated via the following APIs:
     *
     * - `Model::acquireInputBuffer`
     * - `Model::acquireInputBuffers`
     * - `ModelVariantHandle::acquireInputBuffer`
     * - `ModelVariantHandle::acquireInputBuffers`
     *
     * Additionally, `Model::repositionInputs` and `ModelVariantHandle::repositionInputs`
     * must be used properly.
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/
    StatusCode inferBufferToFloat(const std::vector<Buffer>& input,
                                  std::vector<NDArray<float>>& output,
                                  const std::vector<std::vector<int64_t>>& shape = {},
                                  uint32_t cache_size = 0);
    StatusCode inferBufferToFloat(const std::vector<std::vector<Buffer>>& input,
                                  std::vector<NDArray<float>>& output,
                                  const std::vector<std::vector<int64_t>>& shape = {},
                                  uint32_t cache_size = 0);
    StatusCode inferBufferToFloat(const std::vector<Buffer>& input,
                                  std::vector<std::vector<float>>& output,
                                  const std::vector<std::vector<int64_t>>& shape = {},
                                  uint32_t cache_size = 0);
    StatusCode inferBufferToFloat(const std::vector<std::vector<Buffer>>& input,
                                  std::vector<std::vector<float>>& output,
                                  const std::vector<std::vector<int64_t>>& shape = {},
                                  uint32_t cache_size = 0);
    /**@}*/

    /**
     * @brief Development-only API for measuring pure NPU inference speed.
     *
     * Runs NPU inference without uploading inputs and without retrieving outputs.
     *
     * @param[in] variant_idx Index of model variant to run
     * @return A status code indicating the result.
     */
    StatusCode inferSpeedrun(int variant_idx = 0);

    /**
     * @name Asynchronous Inference
     *
     * Performs inference asynchronously.
     *
     * To use asynchronous inference, the model must be created using a `ModelConfig`
     * object with the async pipeline configured to be enabled. This is done by calling
     * @ref ModelConfig::setAsyncPipelineEnabled
     * "ModelConfig::setAsyncPipelineEnabled(true)" before passing the configuration to
     * `Model::create`.
     *
     * Example:
     * @code
     * using namespace mobilint;
     *
     * ModelConfig mc;
     *
     * // Enables support for `inferAsync` and `inferAsyncCHW`
     * mc.setAsyncPipelineEnabled(true);
     *
     * StatusCode sc;
     * std::unique_ptr<Model> model = Model::create("resnet50.mxq", mc, sc);
     * if (!sc) {
     *    // Handle error appropriately
     * }
     *
     * // Now `inferAsync` can be called.
     * Future future = model->inferAsync(input, sc);
     * @endcode
     *
     * @note Functions in the `inferAsync` family (`inferAsync`, `inferAsyncCHW`,
     *       `inferAsyncToFloat`, `inferAsyncCHWToFloat`) typically return immediately.
     *       However, they may block if the input queue in the qbruntime is full.
     *
     * @note For all functions in the `inferAsync` family (`inferAsync`, `inferAsyncCHW`,
     *       `inferAsyncToFloat`, `inferAsyncCHWToFloat`), the data provided through the
     *       `input` parameter must remain unmodified until the asynchronous inference
     *       has completed. Modifying this data during execution may result in invalid
     *       results.
     *
     * @note Currently, only CNN-based models are supported, as asynchronous execution is
     *       particularly effective for this type of workload.
     *
     * @note Limitations:
     *        - RNN/LSTM and LLM models are not supported yet.
     *        - Models requiring CPU offloading are not supported yet.
     *        - Currently, only single-batch inference is supported (i.e., N = 1).
     *        - Currently, Buffer inference is not supported. The following types
     *          are supported in the synchronous API for advanced use cases, but are not
     *          yet available for asynchronous inference:
     *           - Buffer to Buffer
     *           - Buffer to float
     */
    /**@{*/

    /**
     * @brief Initiates asynchronous inference with input in NHWC (batch N, height H,
     *        width W, channels C) or HWC format.
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NHWC or HWC
     *                  format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsync(const std::vector<NDArray<float>>& input, StatusCode& sc);

    /**
     * @brief Initiates asynchronous inference with input in NCHW (batch N, channels C,
     *        height H, width W) or CHW format.
     *
     * @param[in] input A vector of `NDArray<float>`. Each NDArray must be in NCHW or CHW
     *                  format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsyncCHW(const std::vector<NDArray<float>>& input, StatusCode& sc);

    /**
     * @brief This overload supports int8_t-to-int8_t asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<int8_t>`. Each NDArray must be in NHWC or
     *                  HWC format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<int8_t> inferAsync(const std::vector<NDArray<int8_t>>& input, StatusCode& sc);

    /**
     * @brief This overload supports int8_t-to-int8_t asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<int8_t>`. Each NDArray must be in NCHW or
     *                  CHW format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<int8_t> inferAsyncCHW(const std::vector<NDArray<int8_t>>& input,
                                 StatusCode& sc);

    /**
     * @brief This overload supports int8_t-to-float asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<int8_t>`. Each NDArray must be in NHWC or
     *                  HWC format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsyncToFloat(const std::vector<NDArray<int8_t>>& input,
                                    StatusCode& sc);

    /**
     * @brief This overload supports int8_t-to-float asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<int8_t>`. Each NDArray must be in NCHW or
     *                  CHW format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsyncCHWToFloat(const std::vector<NDArray<int8_t>>& input,
                                       StatusCode& sc);

    /**
     * @brief This overload supports uint8_t-to-float asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<uint8_t>`. Each NDArray must be in NHWC or
     *                  HWC format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsync(const std::vector<NDArray<uint8_t>>& input, StatusCode& sc);

    /**
     * @brief This overload supports uint8_t-to-float asynchronous inference.
     *
     * @param[in] input A vector of `NDArray<uint8_t>`. Each NDArray must be in NCHW or
     *                  CHW format.
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference request was successfully
     *                initiated or encountered an error.
     * @return A future that can be used to retrieve the inference result.
     */
    Future<float> inferAsyncCHW(const std::vector<NDArray<uint8_t>>& input,
                                StatusCode& sc);

    /**@}*/

    /**
     * @name Buffer Management APIs
     *
     * These APIs are required when calling `Model::inferBuffer` or
     * `Model::inferBufferToFloat`.
     *
     * Buffers are acquired using:
     * - `acquireInputBuffer`
     * - `acquireOutputBuffer`
     *
     * Any acquired buffer must be released using:
     * - `releaseBuffer`
     * - `releaseBuffers`
     *
     * Repositioning is handled by:
     * - `repositionInputs`
     * - `repositionOutputs`
     *
     * @note These APIs are intended for advanced use rather than typical usage.
     */
    /**@{*/

    // Acquire buffer
    std::vector<Buffer> acquireInputBuffer(
        const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<Buffer> acquireOutputBuffer(
        const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<std::vector<Buffer>> acquireInputBuffers(
        const int batch_size, const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<std::vector<Buffer>> acquireOutputBuffers(
        const int batch_size, const std::vector<std::vector<int>>& seqlens = {}) const;

    // Deallocate acquired Input/Output buffer
    StatusCode releaseBuffer(std::vector<Buffer>& buffer) const;
    StatusCode releaseBuffers(std::vector<std::vector<Buffer>>& buffers) const;

    // Reposition single batch
    StatusCode repositionInputs(const std::vector<float*>& input,
                                std::vector<Buffer>& input_buf,
                                const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionOutputs(const std::vector<Buffer>& output_buf,
                                 std::vector<float*>& output,
                                 const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionOutputs(const std::vector<Buffer>& output_buf,
                                 std::vector<std::vector<float>>& output,
                                 const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionInputs(const std::vector<uint8_t*>& input,
                                std::vector<Buffer>& input_buf,
                                const std::vector<std::vector<int>>& seqlens = {}) const;

    // Reposition multiple batches
    StatusCode repositionInputs(const std::vector<float*>& input,
                                std::vector<std::vector<Buffer>>& input_buf,
                                const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionOutputs(const std::vector<std::vector<Buffer>>& output_buf,
                                 std::vector<float*>& output,
                                 const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionOutputs(const std::vector<std::vector<Buffer>>& output_buf,
                                 std::vector<std::vector<float>>& output,
                                 const std::vector<std::vector<int>>& seqlens = {}) const;
    StatusCode repositionInputs(const std::vector<uint8_t*>& input,
                                std::vector<std::vector<Buffer>>& input_buf,
                                const std::vector<std::vector<int>>& seqlens = {}) const;
    /**@}*/

    /**
     * @brief Returns the total number of model variants available in this model.
     *
     * The `variant_idx` parameter passed to `Model::getModelVariantHandle` must be
     * in the range [0, return value of this function).
     *
     * @return The total number of model variants.
     */
    int getNumModelVariants() const;

    /**
     * @brief Retrieves a handle to the specified model variant.
     *
     * Use the returned `ModelVariantHandle` to query details such as input and output
     * shapes for the selected variant.
     *
     * @param[in] variant_idx Index of the model variant to retrieve.
     *                        Must be in the range [0, getNumModelVariants()).
     * @param[out] sc A reference to a StatusCode variable that will be updated to
     *                indicate success or failure.
     * @return A unique pointer to the corresponding `ModelVariantHandle` if successful;
     *         otherwise, nullptr.
     */
    std::unique_ptr<ModelVariantHandle> getModelVariantHandle(int variant_idx,
                                                              StatusCode& sc) const;

    /**
     * @brief Returns the input shape of the model.
     *
     * @return A reference to the input shape of the model.
     */
    const std::vector<std::vector<int64_t>>& getModelInputShape() const;

    /**
     * @brief Returns the output shape of the model.
     *
     * @return A reference to the output shape of the model.
     */
    const std::vector<std::vector<int64_t>>& getModelOutputShape() const;

    /**
     * @brief Returns the input buffer information for the model.
     *
     * @return A reference to a vector of input buffer information.
     */
    const std::vector<BufferInfo>& getInputBufferInfo() const;

    /**
     * @brief Returns the output buffer information of the model.
     *
     * @return A reference to a vector of output buffer information.
     */
    const std::vector<BufferInfo>& getOutputBufferInfo() const;

    /**
     * @brief Returns the input quantization scale(s) of the model.
     *
     * @return A vector of input scales.
     */
    std::vector<Scale> getInputScale() const;

    /**
     * @brief Returns the output quantization scale(s) of the model.
     *
     * @return A vector of output scales.
     */
    std::vector<Scale> getOutputScale() const;

    /**
     * @brief Returns the model's unique identifier.
     *
     * This identifier distinguishes multiple models within a single user program.
     * It is assigned incrementally, starting from 0 (e.g., 0, 1, 2, 3, ...).
     *
     * @return The model identifier.
     */
    uint32_t getIdentifier() const;

    /**
     * @brief Returns the path to the MXQ model file associated with the Model.
     *
     * @return The MXQ file path.
     */
    std::string getModelPath() const;

    /**
     * @brief Returns informations of KV-cache of the model.
     *
     * @return A vector of CacheInfo objects.
     */
    std::vector<CacheInfo> getCacheInfos() const;

    /**
     * @name KV Cache Management
     *
     * @note These APIs are used for LLM models that utilize KV cache.
     */
    /**@{*/

    /**
     * @brief Dumps the KV cache memory into buffers.
     *
     * Writes the current KV cache data into provided buffers.
     *
     * @param[out] bufs A reference to vectors of byte vectors that will store the KV
     *                  cache data.
     * @return A status code indicating whether the dump operation was successful
     *         or if an error occurred.
     */
    StatusCode dumpCacheMemory(std::vector<std::vector<int8_t>>& bufs);

    /**
     * @brief Dumps the KV cache memory into buffers.
     *
     * Writes the KV cache data into buffers and returns them.
     *
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the dump operation was successful or if an error occurred.
     * @return A vector of byte vectors containing the KV cache data.
     */
    std::vector<std::vector<int8_t>> dumpCacheMemory(StatusCode& sc);

    /**
     * @brief Dumps KV cache memory to files in the specified directory.
     *
     * Writes the KV cache data to binary files within the given directory.
     * Each file is named using the format: `cache_<layer_hash>.bin`.
     *
     * @param[in] cache_dir Path to the directory where KV cache files will be saved.
     * @return A status code indicating whether the dump operation was successful
     *         or if an error occurred.
     */
    StatusCode dumpCacheMemory(const std::string& cache_dir);

    /**
     * @brief Loads the KV cache memory from buffers.
     *
     * Restores the KV cache from the provided buffers.
     *
     * @param[in] bufs A reference to a vector of byte vectors containing the KV cache
     *                 data.
     * @return A status code indicating whether the load operation was successful
     *         or if an error occurred.
     */
    StatusCode loadCacheMemory(const std::vector<std::vector<int8_t>>& bufs);

    /**
     * @brief Loads the KV cache memory from files in the specified directory.
     *
     * Reads KV cache data from files within the given directory and restores them.
     * Each file is named using the format: `cache_<layer_hash>.bin`.
     *
     * @param[in] cache_dir Path to the directory where KV cache files are saved.
     * @return A status code indicating whether the load operation was successful
     *         or if an error occurred.
     */
    StatusCode loadCacheMemory(const std::string& cache_dir);

    /**
     * @brief Filter the tail of the KV cache memory
     *
     * Retains the desired caches in the tail of the KV cache memory, excludes the others,
     * and shifts the remaining caches forward.
     *
     * @param[in] cache_size The number of tokens accumulated in the KV cache so far.
     * @param[in] tail_size The tail size of the KV cache to filter (<=32).
     * @param[in] mask A mask indicating tokens to retain or exclude at the tail of the KV
     *                 cache.
     * @param[out] sc A status code indicating the outcome of the tail filtering.
     * @return New cache size after tail filtering.
     */
    int filterCacheTail(int cache_size, int tail_size, const std::vector<bool>& mask,
                        StatusCode& sc);

    /**
     * @brief Moves the tail of the KV cache memory to the end of the head.
     *
     * Slice the tail of the KV cache memory up to the specified size
     * and moves it to the designated cache position.
     *
     * @param[in] num_head The size of the KV cache head where the tail is appended.
     * @param[in] num_tail The size of the KV cache tail to be moved.
     * @param[in] cache_size The total number of tokens accumulated in the KV cache so
     *                       far.
     * @param[out] sc A status code indicating the result of the tail move.
     * @return The updated cache size after moving the tail.
     */
    int moveCacheTail(int num_head, int num_tail, int cache_size, StatusCode& sc);

    /**@}*/

    /**
     * @name Deprecated APIs
     *
     * @note These APIs are deprecated and should not be used.
     */
    /**@{*/

    /**
     * @deprecated Use `infer(input, output, shape)` instead.
     */
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output, int batch_size);

    /**
     * @deprecated Use `infer(input, shape, sc)` instead.
     */
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          int batch_size, StatusCode& sc);

    /**
     * @deprecated
     */
    uint64_t getLatencyConsumed(const int npu_op_idx) const;

    /**
     * @deprecated
     */
    uint64_t getLatencyFinished(const int npu_op_idx) const;

    /**@}*/

private:
    Model();

#ifndef _MSC_VER
    std::experimental::propagate_const<std::unique_ptr<ModelImpl>> mImpl;
#else
    std::unique_ptr<ModelImpl> mImpl;
#endif

    friend class Accelerator;
};

/**@}*/

}  // namespace mobilint

#endif
