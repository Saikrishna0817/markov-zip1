#pragma once

#include <cstddef>
#include <string>

namespace markov_cero::gpu {

struct DeviceInfo {
    bool available{false};
    int device_id{0};
    std::string name{"None"};
    std::size_t total_memory_bytes{0};
    std::size_t free_memory_bytes{0};
    int compute_capability_major{0};
    int compute_capability_minor{0};
    int warp_size{32};
    int max_threads_per_block{1024};
};

[[nodiscard]] bool is_gpu_available() noexcept;
[[nodiscard]] int get_device_count() noexcept;
[[nodiscard]] DeviceInfo get_device_info(int device_id = 0) noexcept;
void synchronize_device();

} // namespace markov_cero::gpu
