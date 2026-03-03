/**
 * \file
 */

#ifndef QBRUNTIME_MODEL_VARIANT_HANDLE_H_
#define QBRUNTIME_MODEL_VARIANT_HANDLE_H_

#include <stdint.h>

#include <vector>

#include "qbruntime/export.h"
#include "qbruntime/status_code.h"
#include "qbruntime/type.h"

namespace mobilint {

/**
 * \addtogroup CPPAPI
 * @{
 */

class ModelImpl;

/**
 * @brief Handle to a specific variant of a loaded model.
 *
 * This class provides access to variant-specific information such as input/output
 * shapes, buffer information, and quantization scales. It also offers APIs for
 * managing inference buffers, consistent with the interface of the `Model` class.
 *
 * Objects of this class are obtained via `Model::getModelVariantHandle`.
 */
class QBRUNTIME_EXPORT ModelVariantHandle {
public:
    ModelVariantHandle(const ModelVariantHandle& other) = delete;
    ModelVariantHandle(ModelVariantHandle&& other) = delete;
    ModelVariantHandle& operator=(const ModelVariantHandle& rhs) = delete;
    ModelVariantHandle& operator=(ModelVariantHandle&& rhs) noexcept = delete;
    ~ModelVariantHandle();

    /**
     * @brief Returns the index of this model variant.
     *
     * @return Index of the model variant.
     */
    int getVariantIdx() const;

    /**
     * @brief Returns the input shape for this model variant.
     *
     * @return Reference to the input shape.
     */
    const std::vector<std::vector<int64_t>>& getModelInputShape() const;

    /**
     * @brief Returns the output shape for this model variant.
     *
     * @return Reference to the output shape.
     */
    const std::vector<std::vector<int64_t>>& getModelOutputShape() const;

    /**
     * @brief Returns the input buffer information for this variant.
     *
     * @return Reference to a vector of input buffer information.
     */
    const std::vector<BufferInfo>& getInputBufferInfo() const;

    /**
     * @brief Returns the output buffer information for this variant.
     *
     * @return Reference to a vector of output buffer information.
     */
    const std::vector<BufferInfo>& getOutputBufferInfo() const;

    /**
     * @brief Returns the input quantization scale(s) for this variant.
     *
     * @return Vector of input scales.
     */
    std::vector<Scale> getInputScale() const;

    /**
     * @brief Returns the output quantization scale(s) for this variant.
     *
     * @return Vector of output scales.
     */
    std::vector<Scale> getOutputScale() const;

    /**
     * @name Buffer Management APIs
     *
     * These APIs are used when performing inference with `Model::inferBuffer` or
     * `Model::inferBufferToFloat`, using this variant’s input and output shapes.
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
     * @note These APIs are intended for advanced use and follow the same buffer
     *       management interface as the `Model` class.
     */
    /**@{*/

    // Acquire buffer
    std::vector<Buffer> acquireInputBuffer(
        const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<Buffer> acquireOutputBuffer(
        const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<std::vector<Buffer>> acquireInputBuffers(
        int batch_size, const std::vector<std::vector<int>>& seqlens = {}) const;
    std::vector<std::vector<Buffer>> acquireOutputBuffers(
        int batch_size, const std::vector<std::vector<int>>& seqlens = {}) const;

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

private:
    ModelVariantHandle(int variant_idx, const ModelImpl& model_impl);

    const int mIdx;
    const ModelImpl& mModelImpl;

    friend class ModelImpl;
};

}  // namespace mobilint

#endif  // QBRUNTIME_MODEL_VARIANT_HANDLE_H_
