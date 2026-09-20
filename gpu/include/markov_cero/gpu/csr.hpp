#pragma once

#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/linalg/sparse_basis.hpp"
#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::gpu {

class DeviceCsr final {
  public:
    DeviceCsr() noexcept = default;
    DeviceCsr(std::size_t rows, std::size_t cols, std::size_t nnz);

    DeviceCsr(const DeviceCsr&) = delete;
    DeviceCsr& operator=(const DeviceCsr&) = delete;
    DeviceCsr(DeviceCsr&&) noexcept = default;
    DeviceCsr& operator=(DeviceCsr&&) noexcept = default;
    ~DeviceCsr() = default;

    [[nodiscard]] static DeviceCsr from_csc(const linalg::SparseCsc& csc);
    [[nodiscard]] static DeviceCsr from_csc(const model::SparseMatrixCSC& csc);
    [[nodiscard]] static DeviceCsr transpose_from_csc(const linalg::SparseCsc& csc);
    [[nodiscard]] static DeviceCsr transpose_from_csc(const model::SparseMatrixCSC& csc);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t nnz() const noexcept { return nnz_; }

    [[nodiscard]] const DeviceBuffer<std::size_t>& row_offsets() const noexcept {
        return row_offsets_;
    }
    [[nodiscard]] const DeviceBuffer<std::size_t>& col_indices() const noexcept {
        return col_indices_;
    }
    [[nodiscard]] const DeviceBuffer<double>& values() const noexcept {
        return values_;
    }

    [[nodiscard]] DeviceBuffer<std::size_t>& row_offsets() noexcept {
        return row_offsets_;
    }
    [[nodiscard]] DeviceBuffer<std::size_t>& col_indices() noexcept {
        return col_indices_;
    }
    [[nodiscard]] DeviceBuffer<double>& values() noexcept {
        return values_;
    }

    [[nodiscard]] linalg::SparseCsc to_csc_host() const;
    [[nodiscard]] model::SparseMatrixCSC to_model_csc_host() const;
    void download_to(std::vector<std::size_t>& row_offsets,
                     std::vector<std::size_t>& col_indices,
                     std::vector<double>& values) const;

  private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::size_t nnz_{0};
    DeviceBuffer<std::size_t> row_offsets_;
    DeviceBuffer<std::size_t> col_indices_;
    DeviceBuffer<double> values_;
};

} // namespace markov_cero::gpu
