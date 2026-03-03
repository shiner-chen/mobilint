#ifndef MOBILINT_MACCEL_STATUS_CODE_H_
#define MOBILINT_MACCEL_STATUS_CODE_H_

namespace mobilint {

/**
 * \ingroup CPPAPI
 * @{
 */

// clang-format off
// The maximum status code used is 54.
/**
 * @brief Enumerates status codes for the maccel runtime.
 *
 * This enumeration defines return codes used in the maccel runtime to indicate
 * success or specific error conditions. A value of `StatusCode::OK` (0) represents
 * success, while any other value indicates a particular error type.
 */
enum class StatusCode {
    OK = 0,                                      /**< OK */

    InternalError = 23,                          /**< @deprecated Please do not use this. */
    NotImplemented = 18,                         /**< Not implemented */
    BadAlloc = 39,                               /**< Bad allocation */

    Acc_CoreAlreadyInUse = 11,                   /**< Core already in use. */
    Acc_NPUTimeout = 42,                         /**< NPU timeout */
    Acc_NoIMemInitFound = 43,                    /**< No imem initialization found */
    Acc_NoSuchModel = 12,                        /**< No such model */
    Acc_TaskQueueNotFound = 28,                  /**< Task queue not found */

    Driver_FailedToAllocateHostMemory = 13,      /**< Failed to allocate host memory */
    Driver_FailedToAllocateModelMemory = 44,     /**< Failed to allocate model memory */
    Driver_FailedToBuildCmaIoReq = 45,           /**< Failed to build CMA IO request */
    Driver_FailedToClaimCores = 24,              /**< Failed to claim cores */
    Driver_FailedToFreeModelMemory = 46,         /**< Failed to free model memory */
    Driver_FailedToLockCore = 49,                /**< Failed to lock core */
    Driver_FailedToPostInfer = 33,               /**< Failed to post infer */
    Driver_FailedToReadMemoryBuffer = 15,        /**< Failed to read memory buffer */
    Driver_FailedToUnclaimCores = 25,            /**< Failed to unclaim cores */
    Driver_FailedToUnlockCore = 50,              /**< Failed to unlock core */
    Driver_FailedToWaitDone = 34,                /**< Failed to wait done */
    Driver_FailedToWriteMemoryBuffer = 14,       /**< Failed to write memory buffer */
    Driver_NotInitialized = 1,                   /**< Not implemented */
    Driver_WaitDoneTimeout = 35,                 /**< Wait done timeout */
    Driver_WrongBaseAddress = 47,                /**< Wrong base address */

    MemoryPool_AllocatorNotSet = 30,             /**< Allocator not set */
    MemoryPool_BufNotFound = 29,                 /**< Buffer not found */

    Model_BrokenMXQ = 20,                        /**< Broken mxq */
    Model_DtypeMismatched = 22,                  /**< dtype mismatched */
    Model_FailedToAllocMemory = 32,              /**< Failed to allocate memory */
    Model_FailedToLoadMXQ = 3,                   /**< Failed to load mxq */
    Model_FailedToOpenModelDescFile = 19,        /**< Failed to open model description file */
    Model_FailedToOpenOutputScale = 6,           /**< @deprecated Scale file doesn't exist any more. */
    Model_FailedToOpenScaleFile = 4,             /**< @deprecated Scale file doesn't exist any more. */
    Model_FailedToOpenSectionBinary = 8,         /**< Failed to open section binary */
    Model_FailedToSaveTensor = 37,               /**< Failed to save tensor */
    Model_CacheOverflow = 53,                    /**< KV-cache overflow */
    Model_InvalidNPUDtype = 51,                  /**< Invalid NPU dtype */
    Model_InvalidRmemType = 52,                  /**< Invalid RMEM type */
    Model_InvalidOutputScaleValue = 7,           /**< @deprecated Scale file doesn't exist any more. */
    Model_InvalidScaleValue = 5,                 /**< Invalid scale value */
    Model_IsNotPacked = 27,                      /**< @deprecated `Packed` logic doesn't exist any more. */
    Model_IsNotSupportedHardware = 48,           /**< Is not supported hardware */
    Model_IsPacked = 26,                         /**< @deprecated `Packed` logic doesn't exist any more. */
    Model_MXQAndModelConfigNotMatch = 16,        /**< mxq and model config not match */
    Model_NoGlobalCoreWithGlobalMultiMode = 17,  /**< No global core with global multi mode */
    Model_NoTargetCores = 2,                     /**< No target cores */
    Model_NotAlive = 10,                         /**< Not alive */
    Model_NotLaunched = 9,                       /**< Not launched */
    Model_NoCache = 54,                          /**< Model does not support KV-cache */
    Model_PredictError = 36,                     /**< Predict error */
    Model_ShapeMismatched = 21,                  /**< Shape mismatched */
    Model_TaskQueueClosed = 40,                  /**< Task queue closed */
    Model_TaskQueueTimeout = 41,                 /**< Task queue timeout */
    Model_UnexpectedMemoryFormat = 38,           /**< @deprecated */
};
// clang-format on

/**
 * @brief Checks whether the given StatusCode represents an error.
 *
 * This operator enables `StatusCode` to be used in conditional statements for error
 * checking. It returns `false` if `sc` is `StatusCode::OK` (0) and `true` otherwise.
 *
 * @code
 * StatusCode sc = someFunction();
 * if (!sc) { // `!sc` is true when `sc` is not `StatusCode::OK`.
 *     // Handle error
 * }
 * @endcode
 *
 * @param[in] sc The StatusCode to evaluate.
 * @return True if `sc` is an error (nonzero), false if it is `StatusCode::OK`.
 */
inline bool operator!(StatusCode sc) { return static_cast<int>(sc) != 0; }

/**@}*/

}  // namespace mobilint

#endif
