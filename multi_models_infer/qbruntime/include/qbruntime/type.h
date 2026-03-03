// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.
/**
 * \file
 */

#ifndef QBRUNTIME_TYPE_H_
#define QBRUNTIME_TYPE_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "qbruntime/export.h"

namespace mobilint {
/**
 * \defgroup CPPAPI C++ API Reference
 *
 * \brief C++ API provides core functionalities for the NPU.
 * @{
 */

/**
 * @brief Retrieves the version of the qbruntime.
 *
 * @return A string representing the runtime version.
 */
QBRUNTIME_EXPORT std::string getQbRuntimeVersion();

/**
 * @brief Retrieves the Git commit hash of the qbruntime.
 *
 * @return A string containing the Git hash.
 */
QBRUNTIME_EXPORT std::string getQbRuntimeGitVersion();

/**
 * @brief Retrieves the vendor name of the qbruntime.
 *
 * Typically, this function returns "mobilint."
 *
 * @return A string containing the vendor name.
 */
QBRUNTIME_EXPORT std::string getQbRuntimeVendor();

/**
 * @brief Retrieves product information of the qbruntime.
 *
 * This function indicates the product for which the qbruntime is built.
 * For example, it may return values such as "aries2-v4" or "regulus-v4."
 *
 * @return A string containing product details.
 */
QBRUNTIME_EXPORT std::string getQbRuntimeProduct();

/**
 * @brief Enumerates clusters in the ARIES NPU.
 *
 * @note The ARIES NPU consists of two clusters, each containing one global core and
 * four local cores, totaling eight local cores. REGULUS has only a single cluster
 * (Cluster0) with one local core (Core0).
 */
enum class Cluster : int32_t {
    Cluster0 = 1 << 16,  /**< Cluster 0 */
    Cluster1 = 2 << 16,  /**< Cluster 1 */
    Error = 0x7FFF'0000, /**< Represents an invalid or uninitialized state. */
};

/**
 * @brief Enumerates cores within a cluster in the ARIES NPU.
 *
 * @note The ARIES NPU consists of two clusters, each containing one global core and
 * four local cores, totaling eight local cores. REGULUS has only a single cluster
 * (Cluster0) with one local core (Core0).
 */
enum class Core : int32_t {
    Core0 = 1,                /**< Local core 0 */
    Core1 = 2,                /**< Local core 1 */
    Core2 = 3,                /**< Local core 2 */
    Core3 = 4,                /**< Local core 3 */
    All = 0x0000'FFFC,        /**< Deprecated */
    GlobalCore = 0x0000'FFFE, /**< Global core */
    Error = 0x0000'FFFF,      /**< Represents an invalid or uninitialized state. */
};

/**
 * @brief Core allocation policy
 */
enum class CoreAllocationPolicy {
    Auto,   /**< Auto */
    Manual, /**< Manual */
};

/**
 * @brief Struct for scale values.
 */
struct Scale {
    std::vector<float> scale_list; /**< List of scale values for non-uniform scaling. */
    float scale = 0.0F;            /**< Uniform scale value */
    bool is_uniform = false;       /**< Indicates whether uniform scaling is used. */

    /**
     * @brief Returns the scale value at the specified index.
     *
     * @param[in] i Index.
     * @return Scale value.
     */
    float operator[](int i) const {
        if (is_uniform) {
            return scale;
        }
        return scale_list[i];
    }
};

/**
 * @brief Represents a unique identifier for an NPU core.
 *
 * A CoreId consists of a Cluster and a Core, identifying a specific core
 * within an NPU.
 */
struct CoreId {
    Cluster cluster = Cluster::Error; /**< Cluster to which the core belongs. */
    Core core = Core::Error;          /**< Core within the Cluster. */

    /**
     * @brief Checks if two CoreId objects are equal.
     *
     * @param[in] rhs Another CoreId object.
     * @return True if both CoreId objects are identical, false otherwise.
     */
    bool operator==(const CoreId& rhs) const {
        return std::tie(cluster, core) == std::tie(rhs.cluster, rhs.core);
    }

    /**
     * @brief Compares two CoreId objects for ordering.
     *
     * @param[in] rhs Another CoreId object.
     * @return True if this CoreId is less than the given CoreId, false otherwise.
     */
    bool operator<(const CoreId& rhs) const {
        return std::tie(cluster, core) < std::tie(rhs.cluster, rhs.core);
    }
};

/**
 * @brief A simple byte-sized buffer.
 *
 * This struct represents a contiguous block of memory for storing byte-sized data.
 */
struct Buffer {
    int8_t* data = nullptr; /**< Pointer to the data buffer. */
    uint64_t size = 0;      /**< Size of the buffer in bytes. */
};

/**
 * @brief Defines the core mode for NPU execution.
 *
 * Supported core modes include single-core, multi-core, global4-core, and global8-core.
 * For detailed explanations of each mode, refer to the following functions:
 *
 * - `ModelConfig::setSingleCoreMode`
 * - `ModelConfig::setMultiCoreMode`
 * - `ModelConfig::setGlobal4CoreMode`
 * - `ModelConfig::setGlobal8CoreMode`
 */
enum class CoreMode : uint8_t {
    Single = 0,  /**< Single-core mode */
    Multi = 1,   /**< Multi-core mode */
    Global = 2,  /**< Deprecated */
    Global4 = 3, /**< Global4-core mode */
    Global8 = 4, /**< Global8-core mode */
    Error = 0xF, /**< Represents an invalid or uninitialized state. */
};

/**
 * @brief Struct representing input/output buffer information.
 */
struct BufferInfo {
    // clang-format off
    uint32_t original_height = 0;  /**< Height of original input/output */
    uint32_t original_width = 0;   /**< Width of original input/output */
    uint32_t original_channel = 0; /**< Channel of original input/output */
    uint32_t reshaped_height = 0;  /**< Height of reshaped input/output */
    uint32_t reshaped_width = 0;   /**< Width of reshaped input/output */
    uint32_t reshaped_channel = 0; /**< Channel of reshaped input/output */
    uint32_t height = 0;           /**< Height of NPU input/output */
    uint32_t width = 0;            /**< Width of NPU input/output */
    uint32_t channel = 0;          /**< Channel of NPU input/output */
    uint32_t max_height = 0;       /**< Maximum height of original input/output if data is sequential. */
    uint32_t max_width = 0;        /**< Maximum width of original input/output if data is sequential. */
    uint32_t max_channel = 0;      /**< Maximum channel of original input/output if data is sequential. */
    uint32_t max_cache_size = 0;   /**< Maximum KV-cache size, relevant for LLM models using KV cache.*/
    // clang-format on

    /**
     * @brief Returns the total size of the original input/output.
     * @return The data size.
     */
    uint32_t original_size() const {
        return original_height * original_width * original_channel;
    }

    /**
     * @brief Returns the total size of the reshaped input/output.
     * @return The data size.
     */
    uint32_t reshaped_size() const {
        return reshaped_height * reshaped_width * reshaped_channel;
    }

    /**
     * @brief Returns the total size of the NPU input/output.
     * @return The data size.
     */
    uint32_t size() const { return height * width * channel; }
};

/**
 * @brief Configures a core mode and core allocation of a model for NPU inference.
 *
 * The `ModelConfig` class provides methods for setting a core mode and allocating
 * cores for NPU inference. Supported core modes are single-core, multi-core,
 * global4-core, and global8-core. Users can also specify which cores to allocate for
 * the model. Additionally, the configuration offers an option to enforce the use of a
 * specific NPU bundle.
 *
 * @note Deprecated functions are included for backward compatibility, but it is
 * recommended to use the newer core mode configuration methods.
 */
class QBRUNTIME_EXPORT ModelConfig {
public:
    /**
     * @brief Default constructor. This default-constructed object is initially set to
     * single-core mode with all NPU local cores included.
     */
    ModelConfig();

    /**
     * @brief Sets the model to use single-core mode for inference with a specified number
     * of local cores.
     *
     * In single-core mode, each local core executes model inference independently.
     * The number of cores used is specified by the `num_cores` parameter, and the core
     * allocation policy is set to `CoreAllocationPolicy::Auto`, meaning the model will be
     * automatically allocated to available local cores when the model is launched to the
     * NPU, specifically when the `Model::launch` function is called.
     *
     * @param[in] num_cores The number of local cores to use for inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setSingleCoreMode(int num_cores);

    /**
     * @brief Sets the model to use single-core mode for inference with a specific set of
     * NPU local cores.
     *
     * In single-core mode, each local core executes model inference independently.
     * The user can specify a vector of CoreIds to determine which cores to use for
     * inference.
     *
     * @param[in] core_ids A vector of CoreIds to be used for model inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setSingleCoreMode(std::vector<CoreId> core_ids);

    /**
     * @brief Sets the model to use multi-core mode for batch inference.
     *
     * In multi-core mode, on Aries NPU, the four local cores within a cluster work
     * together to process batch inference tasks efficiently. This mode is optimized for
     * batch processing.
     *
     * @note By default, the configuration is set to use all clusters.
     *
     * @param[in] clusters A vector of clusters to be used for multi-core batch inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setMultiCoreMode(std::vector<Cluster> clusters = {Cluster::Cluster0,
                                                           Cluster::Cluster1});

    /**
     * @brief Sets the model to use global4-core mode for inference with a specified set
     * of NPU clusters.
     *
     * For Aries NPU, there are two clusters, each consisting of four local cores. In
     * global4-core mode, four local cores within the same cluster work together to
     * execute the model inference.
     *
     * @note By default, the configuration is set to use all clusters.
     *
     * @param[in] clusters A vector of clusters to be used for model inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setGlobal4CoreMode(std::vector<Cluster> clusters = {Cluster::Cluster0,
                                                             Cluster::Cluster1});

    /**
     * @brief Sets the model to use global8-core mode for inference.
     *
     * For Aries NPU, there are two clusters, each consisting of four local cores. In
     * global8-core mode, all eight local cores across the two clusters work together to
     * execute the model inference.
     *
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setGlobal8CoreMode();

    /**
     * @brief Gets the core mode to be applied to the model.
     *
     * This reflects the core mode that will be used when the model is created.
     *
     * @return The `CoreMode` to be applied to the model.
     */
    CoreMode getCoreMode() const { return mCoreMode; }

    /**
     * @brief Gets the core allocation policy to be applied to the model.
     *
     * This reflects the core allocation policy that will be used when the model is
     * created.
     *
     * @return The `CoreAllocationPolicy` to be applied to the model.
     */
    CoreAllocationPolicy getCoreAllocationPolicy() const { return mCoreAllocationPolicy; }

    /**
     * @brief Gets the number of cores to be allocated for the model.
     *
     * This represents the number of cores that will be allocated for inference
     * when the model is launched to the NPU.
     *
     * @return The number of cores to be allocated for the model.
     */
    int getNumCores() const { return mNumCores; }

    /**
     * @brief Forces the use of a specific NPU bundle.
     *
     * This function forces the selection of a specific NPU bundle. If a non-negative
     * index is provided, the corresponding NPU bundle is selected and runs without CPU
     * offloading. If -1 is provided, all NPU bundles are used with CPU offloading
     * enabled.
     *
     * @param[in] npu_bundle_index The index of the NPU bundle to force. A non-negative
     *                             integer selects a specific NPU bundle (runs without CPU
     *                             offloading), or -1 to enable all NPU bundles with CPU
     *                             offloading.
     * @return true if the index is valid and the NPU bundle is successfully set,
     *         false if the index is invalid (less than -1).
     */
    bool forceSingleNPUBundle(int npu_bundle_index);

    /**
     * @brief Retrieves the index of the forced NPU bundle.
     *
     * This function returns the index of the NPU bundle that has been forced using the
     * `forceSingleNPUBundle` function. If no NPU bundle is forced, the returned value
     * will be -1.
     *
     * @return The index of the forced NPU bundle, or -1 if no bundle is forced.
     */
    int getForcedNPUBundleIndex() const { return mForcedNPUBundleIndex; }

    /**
     * @brief Returns the list of NPU CoreIds to be used for model inference.
     *
     * This function returns a reference to the vector of NPU CoreIds that the model
     * will use for inference. When setSingleCoreMode(int num_cores) is called and the
     * core allocation policy is set to CoreAllocationPolicy::Auto, it will return an
     * empty vector.
     *
     * @return A constant reference to the vector of NPU CoreIds.
     */
    const std::vector<CoreId>& getCoreIds() const { return mCoreIds; }

    const std::vector<Cluster>& getClusters() const { return mClusters; }

    /**
     * @brief Enables or disables the asynchronous pipeline required for asynchronous
     *        inference.
     *
     * Call this function with `enable` set to `true` if you intend to use
     * `Model::inferAsync` or `Model::inferAsyncCHW`, as the asynchronous pipeline is
     * necessary for their operation.
     *
     * If you are only using synchronous inference, such as `Model::infer` or
     * `Model::inferCHW`, it is recommended to keep the asynchronous pipeline disabled
     * to avoid unnecessary overhead.
     *
     * @param[in] enable Set to `true` to enable the asynchronous pipeline; set to `false`
     *                   to disable it.
     */
    void setAsyncPipelineEnabled(bool enable);

    /**
     * @brief Returns whether the asynchronous pipeline is enabled in this configuration.
     *
     * @return `true` if the asynchronous pipeline is enabled; `false` otherwise.
     */
    bool getAsyncPipelineEnabled() const { return mAsyncPipelineEnabled; }

    /**
     * @brief Sets activation buffer slots for multi-activation supported model.
     *
     * Call this function if you want to set the number of activation buffer slots
     * manually.
     *
     * If you do not call this function, the default number of activation buffer slots
     * is set differently depending on the CoreMode.
     *
     * - `CoreMode::Single` : 2 * (the number of target core ids)
     * - `CoreMode::Multi` : 2 * (the number of target clusters)
     * - `CoreMode::Global4` : 2 * (the number of target clusters)
     * - `CoreMode::Global8` : 2
     *
     * @note This function has no effect on MXQ file in version earlier than MXQv7.
     *
     * @note Currently, LLM model's activation slot is fixed to 1 and ignoring `count`.
     *
     * @param[in] count Multi activation counts. Must be >= 1.
     */
    void setActivationSlots(int count);

    /**
     * @brief Returns activation buffer slot count.
     *
     * @note This function has no meaning on MXQ file in version earlier than MXQv7.
     *
     * @return Activation buffer slot count.
     */
    int getActivationSlots() const { return mActivationSlots; }

    explicit ModelConfig(int num_cores); /**< deprecated */

    bool setGlobalCoreMode(std::vector<Cluster> clusters); /**< deprecated */

    /**
     * @deprecated This setting has no effect.
     */
    std::vector<uint64_t> early_latencies;
    /**
     * @deprecated This setting has no effect.
     */
    std::vector<uint64_t> finish_latencies;

private:
    CoreMode mCoreMode = CoreMode::Single;
    CoreAllocationPolicy mCoreAllocationPolicy = CoreAllocationPolicy::Manual;
    std::vector<Cluster> mClusters;
    std::vector<CoreId> mCoreIds;
    int mNumCores;
    int mForcedNPUBundleIndex = -1;  // -1 means single npu bundle usage is not forced.
    bool mAsyncPipelineEnabled = false;
    int mActivationSlots = 1;
};

/**
 * @brief LogLevel
 */
enum class LogLevel : char {
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERR = 4,
    FATAL = 5,
    OFF = 6,
};

/**
 * @brief CacheType
 */
enum class CacheType : uint8_t { Default = 0, Batch, Error = 0x0F };

/**
 * @brief Struct representing KV-cache information.
 */
struct CacheInfo {
    CacheType cache_type = CacheType::Error;
    std::string name;
    std::string layer_hash;
    uint64_t size = 0;
    size_t num_batches = 0;
};

QBRUNTIME_EXPORT void setLogLevel(LogLevel level);

/**
 * @brief Starts event tracing and prepares to save the trace log to a specified file.
 *
 * The trace log is recorded in "Chrome Tracing JSON format," which can be
 * viewed at https://ui.perfetto.dev/.
 *
 * The trace log is not written immediately; it is saved only when
 * stopTracingEvents() is called.
 *
 * @param[in] path The file path where the trace log should be stored.
 * @return True if tracing starts successfully, false otherwise.
 */
QBRUNTIME_EXPORT bool startTracingEvents(const char* path);

/**
 * @brief Stops event tracing and writes the recorded trace log.
 *
 * This function finalizes tracing and saves the collected trace data
 * to the file specified when startTracingEvents() was called.
 */
QBRUNTIME_EXPORT void stopTracingEvents();

/**
 * @brief Generates a structured summary of the specified MXQ model.
 *
 * Returns an overview of the model contained in the MXQ file, including:
 * - Target NPU hardware
 * - Supported core modes and their associated cores
 * - The total number of model variants
 * - For each variant:
 *   - Input and output tensor shapes
 *   - A list of layers with their types, output shapes, and input layer indices
 *
 * The summary is returned as a human-readable string in a table and is useful for
 * inspecting model compatibility, structure, and input/output shapes.
 *
 * @param[in] mxq_path Path to the MXQ model file.
 * @return A formatted string containing the model summary.
 */
QBRUNTIME_EXPORT std::string getModelSummary(const std::string& mxq_path);

/**@}*/

}  // namespace mobilint

#endif
