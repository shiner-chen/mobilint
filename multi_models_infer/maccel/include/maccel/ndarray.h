// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_NDARRAY_H_
#define MACCEL_NDARRAY_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "maccel/export.h"
#include "maccel/status_code.h"
#include "maccel/type.h"

namespace mobilint {
/**
 * \addtogroup CPPAPI
 * @{
 */
namespace internal {

inline int64_t numel(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), INT64_C(1), std::multiplies<>());
}

/**
 * @brief A class representing the data underlying an NDArray.
 *
 * This class is responsible for holding the raw memory buffer or pointer for
 * an NDArray. It is an internal implementation detail and is not intended for
 * direct use by external code.
 *
 * **Do not use this class directly; please use NDArray instead.**
 */
class MACCEL_EXPORT NDArrayData {
public:
    NDArrayData() = default;
    NDArrayData(int64_t bytesize, StatusCode& sc);
    explicit NDArrayData(std::shared_ptr<void> data)
        : mData(std::move(data)), mRawPtr(mData.get()) {}
    explicit NDArrayData(void* ptr) : mData{nullptr}, mRawPtr{ptr} {};

    NDArrayData(const NDArrayData& other) = default;
    NDArrayData(NDArrayData&& other) noexcept = default;
    NDArrayData& operator=(const NDArrayData& other);
    NDArrayData& operator=(NDArrayData&& other) noexcept = default;

    void* data() const { return mRawPtr; }

private:
    std::shared_ptr<void> mData = nullptr;  // Owned data
    void* mRawPtr = nullptr;                // A raw pointer, which is not owned
};

}  // namespace internal

/**
 * @brief A class representing an N-dimensional array (NDArray).
 *
 * The class automatically manages memory when it owns the data. If the NDArray is created
 * from a raw pointer without taking ownership, it simply acts as a view of the existing
 * data, and no memory management is performed by the NDArray.
 *
 * `NDArray` is the recommended way for providing input and receiving output to/from
 * model inferences. By using `NDArray`, unnecessary copies inside the maccel runtime
 * are avoided, which helps improve performance during model inference operations.
 *
 * @tparam T The type of the elements in the array.
 */
template <typename T>
class NDArray {
public:
    using value_type = T;

    /**
     * @brief Default constructor. Creates an empty NDArray.
     */
    NDArray() = default;

    /**
     * @brief Constructs an NDArray with the given shape, allocating memory for elements.
     *
     * This constructor allocates memory for the array and takes ownership of the
     * allocated data. The memory is automatically managed by the NDArray instance.
     *
     * @param[in] shape The shape (dimensions) of the array.
     * @param[out] sc Status code indicating success or failure of memory allocation.
     */
    NDArray(const std::vector<int64_t>& shape, StatusCode& sc)
        : mSize(internal::numel(shape)), mData(sizeof(T) * mSize, sc), mShape(shape) {}

    /**
     * @brief Constructs an NDArray from a raw pointer without taking ownership.
     *
     * This constructor creates an NDArray that acts as a view of the provided data.
     * It does not manage the memory, and the memory is not owned by the NDArray instance.
     * The user is responsible for ensuring that the data remains valid for the lifetime
     * of the NDArray.
     *
     * @param[in] ptr A pointer to the data.
     * @param[in] shape The shape (dimensions) of the array.
     */
    NDArray(T* ptr, const std::vector<int64_t>& shape)
        : mSize(internal::numel(shape)), mData(ptr), mShape(shape) {}

    /**
     * @brief Constructs an NDArray from a raw pointer without taking ownership.
     *
     * @deprecated This constructor is deprecated.
     *             Use `NDArray(T*, const std::vector<int64_t>&)` instead.
     *
     * @param[in] ptr A pointer to the data.
     * @param[in] shape The shape (dimensions) of the array.
     * @param owner Ignored parameter (was intended to indicate ownership).
     */
    NDArray(T* ptr, const std::vector<int64_t>& shape, bool owner)
        : NDArray(ptr, shape) {}

    /**
     * @brief Constructs an NDArray from a shared memory block with shared ownership.
     *
     * This constructor allows the NDArray to share ownership of an existing memory
     * buffer, preventing unnecessary copies. The `data` parameter should be a
     * `std::shared_ptr<void>` pointing to a valid memory block.
     *
     * Example usage:
     * @code
     * size_t bytesize = sizeof(float) * 100; // Allocate memory for 100 floats
     * void* raw_ptr = std::malloc(bytesize);
     * if (!raw_ptr) {
     *     // Handle memory allocation failure
     * }
     * auto buffer = std::shared_ptr<void>(raw_ptr, std::free);
     * NDArray<float> array(buffer, {10, 10}); // Create a 10x10 array
     * @endcode
     *
     * @param[in] data A shared pointer to the memory buffer.
     *                 For example, it can be initialized using
     *                 `std::shared_ptr<void>(std::malloc(bytesize), std::free)`,
     *                 but ensure that the allocation succeeds before passing it.
     * @param[in] shape The shape (dimensions) of the array.
     */
    NDArray(std::shared_ptr<void> data, const std::vector<int64_t>& shape)
        : mSize(internal::numel(shape)), mData(std::move(data)), mShape(shape) {}

    /**
     * @brief Returns a pointer to the underlying data.
     *
     * @return A pointer to the array's data.
     */
    T* data() const { return reinterpret_cast<T*>(mData.data()); }

    /**
     * @brief Returns the shape (dimensions) of the array.
     *
     * @return A reference to the vector representing the shape.
     */
    const std::vector<int64_t>& shape() const { return mShape; }

    /**
     * @brief Returns the number of dimensions of the array.
     *
     * @return The number of dimensions.
     */
    std::size_t ndim() const { return mShape.size(); }

    /**
     * @brief Returns the total number of elements in the array.
     *
     * @return The total number of elements.
     */
    int64_t size() const { return mSize; }

    /**
     * @brief Returns an iterator to the beginning of the data.
     *
     * @return A pointer to the first element.
     */
    T* begin() { return data(); }

    /**
     * @brief Returns an iterator to the end of the data.
     *
     * @return A pointer past the last element.
     */
    T* end() { return data() + mSize; }

    /**
     * @brief Returns a constant iterator to the beginning of the data.
     *
     * @return A pointer to the first element.
     */
    const T* begin() const { return data(); }

    /**
     * @brief Returns a constant iterator to the end of the data.
     *
     * @return A pointer past the last element.
     */
    const T* end() const { return data() + mSize; }

    /**
     * @brief Accesses an element of the array.
     *
     * @param idx The index of the element.
     * @return A reference to the element at the specified index.
     */
    T& operator[](size_t idx) { return data()[idx]; }

    /**
     * @brief Accesses an element of the array (const version).
     *
     * @param idx The index of the element.
     * @return A copy of the element at the specified index.
     */
    T operator[](size_t idx) const { return data()[idx]; }

private:
    int64_t mSize = 0;
    internal::NDArrayData mData;
    std::vector<int64_t> mShape;
};

/**@}*/

}  // namespace mobilint

#endif  // MACCEL_NDARRAY_H_
