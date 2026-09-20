#include "markov_cero/gpu/buffer.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#ifdef MARKOV_CERO_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace markov_cero::gpu {

namespace detail {

void* allocate_device_memory(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, bytes);
    if (err != cudaSuccess || dev_ptr == nullptr) {
        throw std::bad_alloc();
    }
    return dev_ptr;
#else
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
#endif
}

void free_device_memory(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
#ifdef MARKOV_CERO_HAS_CUDA
    cudaFree(ptr);
#else
    std::free(ptr);
#endif
}

void copy_host_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument("copy_host_to_device encountered null pointer");
    }
#ifdef MARKOV_CERO_HAS_CUDA
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") +
                                 cudaGetErrorString(err));
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

void copy_device_to_host(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument("copy_device_to_host encountered null pointer");
    }
#ifdef MARKOV_CERO_HAS_CUDA
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") +
                                 cudaGetErrorString(err));
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

void copy_device_to_device(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument("copy_device_to_device encountered null pointer");
    }
#ifdef MARKOV_CERO_HAS_CUDA
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2D failed: ") +
                                 cudaGetErrorString(err));
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

} // namespace detail

template class DeviceBuffer<double>;
template class DeviceBuffer<std::size_t>;
template class DeviceBuffer<int>;

} // namespace markov_cero::gpu
