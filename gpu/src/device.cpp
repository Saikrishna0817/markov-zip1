#include "markov_cero/gpu/device.hpp"

#ifdef MARKOV_CERO_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace markov_cero::gpu {

bool is_gpu_available() noexcept {
#ifdef MARKOV_CERO_HAS_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

int get_device_count() noexcept {
#ifdef MARKOV_CERO_HAS_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count < 0) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

DeviceInfo get_device_info(int device_id) noexcept {
    DeviceInfo info;
#ifdef MARKOV_CERO_HAS_CUDA
    int count = get_device_count();
    if (device_id < 0 || device_id >= count) {
        return info;
    }

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
        return info;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_bytes, &total_bytes);

    info.available = true;
    info.device_id = device_id;
    info.name = props.name;
    info.total_memory_bytes = total_bytes;
    info.free_memory_bytes = free_bytes;
    info.compute_capability_major = props.major;
    info.compute_capability_minor = props.minor;
    info.warp_size = props.warpSize;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    return info;
#else
    (void)device_id;
    info.name = "CPU Fallback (No CUDA)";
    return info;
#endif
}

void synchronize_device() {
#ifdef MARKOV_CERO_HAS_CUDA
    cudaDeviceSynchronize();
#endif
}

} // namespace markov_cero::gpu
