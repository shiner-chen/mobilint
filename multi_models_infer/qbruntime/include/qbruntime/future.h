/**
 * \file
 */

#ifndef QBRUNTIME_FUTURE_H_
#define QBRUNTIME_FUTURE_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "qbruntime/export.h"
#include "qbruntime/ndarray.h"
#include "qbruntime/type.h"

namespace mobilint {

/**
 * \addtogroup CPPAPI
 * @{
 */

template <typename T>
class FutureImpl;

/**
 * @brief Represents a future for retrieving the result of asynchronous inference.
 *
 * This class provides a mechanism similar to std::future, allowing access to the result
 * of an asynchronous inference operation initiated via Model::inferAsync or
 * Model::inferAsyncCHW.
 *
 * The Future object enables the caller to:
 * - Wait for the inference to complete (`waitFor`)
 * - Block until completion and retrieve the output (`get`)
 *
 * Like std::future, this object can be moved but cannot be copied. It is associated with
 * a unique asynchronous result. Once the output is retrieved via `get`, the Future
 * becomes invalid and should not be reused.
 */
template <typename T>
class QBRUNTIME_EXPORT Future {
public:
    Future();
    ~Future();

    Future(const Future& other) = delete;
    Future& operator=(const Future& other) = delete;
    Future(Future&& other) noexcept;
    Future& operator=(Future&& other) noexcept;

    /**
     * @brief Waits for asynchronous inference to complete or until the timeout elapses.
     *
     * @note This method is safe to call multiple times. Calling it with
     *       a timeout of zero (i.e., `waitFor(0)`) performs a non-blocking
     *       check to see whether asynchronous inference has already completed.
     *
     * @param[in] timeout Maximum duration to wait, in milliseconds.
     * @return True if inference completed before the timeout, false otherwise.
     */
    bool waitFor(int64_t timeout_ms);

    /**
     * @brief Blocks until asynchronous inference completes and retrieves the output.
     *
     * @note This method should be called only once per Future. If called again,
     *       the return value will be invalid, and the status code will be set to
     *       StatusCode::Future_NotValid.
     *
     * @param[out] sc A reference to a status code that will be updated to indicate
     *                whether the asynchronous inference completed successfully or if an
     *                error occurred.
     * @return A vector of NDArray<T> containing the inference output.
     */
    std::vector<NDArray<T>> get(StatusCode& sc);

private:
    Future(std::unique_ptr<FutureImpl<T>> future_impl);

    std::unique_ptr<FutureImpl<T>> mImpl;
    bool mOwner = false;

    friend class FutureImpl<T>;
};

/**@}*/

}  // namespace mobilint

#endif
