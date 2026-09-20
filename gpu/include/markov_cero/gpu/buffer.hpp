#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace markov_cero::gpu {

namespace detail {

[[nodiscard]] void* allocate_device_memory(std::size_t bytes);
void free_device_memory(void* ptr) noexcept;
void copy_host_to_device(void* dst, const void* src, std::size_t bytes);
void copy_device_to_host(void* dst, const void* src, std::size_t bytes);
void copy_device_to_device(void* dst, const void* src, std::size_t bytes);

} // namespace detail

template <typename T>
class DeviceBuffer final {
  public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    DeviceBuffer(const T* host_src, std::size_t count) {
        allocate(count);
        upload(host_src, count);
    }

    explicit DeviceBuffer(const std::vector<T>& host_vec) {
        allocate(host_vec.size());
        upload(host_vec.data(), host_vec.size());
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count == size_ && (count == 0 || data_ != nullptr)) {
            return;
        }
        release();
        if (count > 0) {
            data_ = static_cast<T*>(detail::allocate_device_memory(count * sizeof(T)));
            size_ = count;
        }
    }

    void release() noexcept {
        if (data_ != nullptr) {
            detail::free_device_memory(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    void upload(const T* host_src, std::size_t count) {
        if (count == 0) {
            return;
        }
        if (host_src == nullptr) {
            throw std::invalid_argument("DeviceBuffer::upload null source pointer");
        }
        if (count > size_ || data_ == nullptr) {
            throw std::invalid_argument("DeviceBuffer::upload count exceeds buffer size");
        }
        detail::copy_host_to_device(data_, host_src, count * sizeof(T));
    }

    void upload(const std::vector<T>& host_vec) {
        upload(host_vec.data(), host_vec.size());
    }

    void download(T* host_dst, std::size_t count) const {
        if (count == 0) {
            return;
        }
        if (host_dst == nullptr) {
            throw std::invalid_argument("DeviceBuffer::download null destination pointer");
        }
        if (count > size_ || data_ == nullptr) {
            throw std::invalid_argument("DeviceBuffer::download count exceeds buffer size");
        }
        detail::copy_device_to_host(host_dst, data_, count * sizeof(T));
    }

    void download(std::vector<T>& host_vec) const {
        host_vec.resize(size_);
        download(host_vec.data(), size_);
    }

    [[nodiscard]] std::vector<T> to_vector() const {
        std::vector<T> result(size_);
        download(result.data(), size_);
        return result;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  private:
    T* data_{nullptr};
    std::size_t size_{0};
};

} // namespace markov_cero::gpu
