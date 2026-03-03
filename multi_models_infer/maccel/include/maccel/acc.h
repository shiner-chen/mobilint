// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_ACC_H_
#define MACCEL_ACC_H_

#ifndef _MSC_VER
#include <experimental/propagate_const>
#endif
#include <memory>
#include <string>
#include <vector>

#include "maccel/export.h"
#include "maccel/model.h"
#include "maccel/status_code.h"
#include "maccel/type.h"

namespace mobilint {

/**
 * \addtogroup CPPAPI
 * @{
 */

class AcceleratorImpl;

/**
 * @brief Retrieves the version of the maccel runtime.
 *
 * @return A string representing the runtime version.
 */
MACCEL_EXPORT const std::string& getMaccelVersion();

/**
 * @brief Retrieves the Git commit hash of the maccel runtime.
 *
 * @return A string containing the Git hash.
 */
MACCEL_EXPORT const std::string& getMaccelGitVersion();

/**
 * @brief Retrieves the vendor name of the maccel runtime.
 *
 * Typically, this function returns "mobilint."
 *
 * @return A string containing the vendor name.
 */
MACCEL_EXPORT const std::string& getMaccelVendor();

/**
 * @brief Retrieves product information of the maccel runtime.
 *
 * This function indicates the product for which the maccel runtime is built.
 * For example, it may return values such as "aries2-v4" or "regulus-v4."
 *
 * @return A string containing product details.
 */
MACCEL_EXPORT const std::string& getMaccelProduct();

/**
 * @brief Represents an accelerator, i.e., an NPU, used for executing models.
 */
class MACCEL_EXPORT Accelerator {
public:
    /**
     * @brief Creates an Accelerator object for executing models, using device number 0.
     *
     * This function is useful when only one device is attached, in which case the default
     * device (device 0) will be used.
     *
     * @param[out] sc Reference to a StatusCode that will be updated with the result of
     *                the creation.
     * @return A unique pointer to the created Accelerator object.
     */
    static std::unique_ptr<Accelerator> create(StatusCode& sc);

    /**
     * @brief Creates an Accelerator object for a specific device.
     *
     * The `dev_no` parameter represents the device number. For example, on Linux,
     * if an ARIES NPU is attached as `/dev/aries0`, the device number is `0`.
     *
     * @param dev_no The device number to associate with the Accelerator.
     * @param[out] sc Reference to a StatusCode that will be updated with the result of
     *                the creation.
     * @return A unique pointer to the created Accelerator object.
     */
    static std::unique_ptr<Accelerator> create(int dev_no, StatusCode& sc);

    /**
     * @deprecated Use `std::unique_ptr<Accelerator> create(int dev_no, StatusCode& sc)`
     * instead.
     */
    static std::unique_ptr<Accelerator> create(int dev_no, bool verbose, StatusCode& sc);

    Accelerator(const Accelerator& other) = delete;
    Accelerator(Accelerator&& other) noexcept;
    Accelerator& operator=(const Accelerator& rhs) = delete;
    Accelerator& operator=(Accelerator&& rhs) noexcept;

    /**
     * @brief Destroys the Accelerator object and releases resources.
     */
    ~Accelerator();

    /**
     * @brief Retrieves a list of available NPU cores.
     *
     * An available core is one that can be allocated for newly created Model objects.
     *
     * @note Availability checks are only supported on Linux. On Windows, this function
     *       returns all NPU cores without checking availability.
     *
     * @return A vector containing the IDs of available cores.
     */
    std::vector<CoreId> getAvailableCores() const;

    /**
     * @brief Starts event tracing and prepares to save the trace log to a specified file.
     *
     * The trace log is recorded in "Chrome Tracing JSON format," which can be
     * viewed at https://ui.perfetto.dev/.
     *
     * The trace log is not written immediately; it is saved only when
     * stopTracingEvents() is called.
     *
     * @deprecated Use mobilint::startTracingEvents(const char*) instead.
     *
     * @param[in] path The file path where the trace log should be stored.
     * @return True if tracing starts successfully, false otherwise.
     */
    bool startTracingEvents(const char* path);

    /**
     * @brief Stops event tracing and writes the recorded trace log.
     *
     * This function finalizes tracing and saves the collected trace data
     * to the file specified when startTracingEvents() was called.
     *
     * @deprecated Use mobilint::stopTracingEvents() instead.
     */
    void stopTracingEvents();

private:
    Accelerator();

#ifndef _MSC_VER
    std::experimental::propagate_const<std::unique_ptr<AcceleratorImpl>> mImpl;
#else
    std::unique_ptr<AcceleratorImpl> mImpl;
#endif

    friend class Model;
};

/**@}*/

}  // namespace mobilint

#endif
