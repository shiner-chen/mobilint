// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.
/**
 * \file
 */

#ifndef QBRUNTIME_ACC_H_
#define QBRUNTIME_ACC_H_

#ifndef _MSC_VER
#include <experimental/propagate_const>
#endif
#include <memory>
#include <string>
#include <vector>

#include "qbruntime/export.h"
#include "qbruntime/model.h"
#include "qbruntime/status_code.h"
#include "qbruntime/type.h"

namespace mobilint {

/**
 * \addtogroup CPPAPI
 * @{
 */

class AcceleratorImpl;

/**
 * @brief Represents an accelerator, i.e., an NPU, used for executing models.
 */
class QBRUNTIME_EXPORT Accelerator {
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

private:
    Accelerator();

#ifndef _MSC_VER
    std::experimental::propagate_const<std::unique_ptr<AcceleratorImpl>> mImpl;
#else
    std::unique_ptr<AcceleratorImpl> mImpl;
#endif

    friend class Model;
    friend class AsyncModel;
};

/**@}*/

}  // namespace mobilint

#endif
