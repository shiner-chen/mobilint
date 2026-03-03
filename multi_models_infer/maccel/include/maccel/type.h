// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_TYPE_H_
#define MACCEL_TYPE_H_

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "maccel/export.h"

namespace mobilint {
/**
 * \defgroup CPPAPI C++ API Reference
 *
 * \brief C++ API provides core functionalities for the NPU.
 * @{
 */

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
 * @deprecated This enum is deprecated.
 */
enum class CollaborationModel {
    Unified,
    Separated,
    Undefined,
};

/**
 * @deprecated This enum is deprecated.
 */
enum class CoreStatus {
    Vacant,
    Ready,
    Idle,
    Running,
};

/**
 * @deprecated This enum is deprecated.
 */
enum class SchedulePolicy {
    FIFO,
    LIFO,
    ByPriority,
    Undefined,
};

/**
 * @deprecated This enum is deprecated.
 */
enum class LatencySetPolicy {
    Auto,
    Manual,
};

/**
 * @deprecated This enum is deprecated.
 */
enum class MaintenancePolicy {
    Maintain,
    DropExpired,
    Undefined,
};

/**
 * @deprecated This enum is deprecated.
 */
enum class InferenceResult {
    Successful,
    Expired,
    Unexpected,
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

class Statistics;

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
class MACCEL_EXPORT ModelConfig {
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
     * @param[in] clusters A vector of clusters to be used for multi-core batch inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setMultiCoreMode(std::vector<Cluster> clusters);

    /**
     * @brief Sets the model to use global4-core mode for inference with a specified set
     * of NPU clusters.
     *
     * For Aries NPU, there are two clusters, each consisting of four local cores. In
     * global4-core mode, four local cores within the same cluster work together to
     * execute the model inference.
     *
     * @param[in] clusters A vector of clusters to be used for model inference.
     * @return true if the mode was successfully set, false otherwise.
     */
    bool setGlobal4CoreMode(std::vector<Cluster> clusters);

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

    explicit ModelConfig(int num_cores); /**< deprecated */

    bool includeAllCores();                   /**< deprecated */
    bool excludeAllCores();                   /**< deprecated */
    bool include(Cluster cluster, Core core); /**< deprecated */
    bool include(Cluster cluster);            /**< deprecated */
    bool include(Core core);                  /**< deprecated */

    bool exclude(Cluster cluster, Core core); /**< deprecated */
    bool exclude(Cluster cluster);            /**< deprecated */
    bool exclude(Core core);                  /**< deprecated */

    bool setGlobalCoreMode(std::vector<Cluster> clusters); /**< deprecated */

    bool setAutoMode(int num_cores = 1); /**< deprecated */
    bool setManualMode();                /**< deprecated */

    /**
     * @deprecated This setting has no effect.
     */
    SchedulePolicy schedule_policy = SchedulePolicy::FIFO;
    /**
     * @deprecated This setting has no effect.
     */
    LatencySetPolicy latency_set_policy = LatencySetPolicy::Auto;
    /**
     * @deprecated This setting has no effect.
     */
    MaintenancePolicy maintenance_policy = MaintenancePolicy::Maintain;
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
    std::vector<CoreId> mCoreIds;
    int mNumCores;
    int mForcedNPUBundleIndex = -1;  // -1 means single npu bundle usage is not forced.
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

MACCEL_EXPORT void setLogLevel(LogLevel level);

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
MACCEL_EXPORT bool startTracingEvents(const char* path);

/**
 * @brief Stops event tracing and writes the recorded trace log.
 *
 * This function finalizes tracing and saves the collected trace data
 * to the file specified when startTracingEvents() was called.
 */
MACCEL_EXPORT void stopTracingEvents();

/**@}*/

}  // namespace mobilint

#endif
