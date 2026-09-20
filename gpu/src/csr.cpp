#include "markov_cero/gpu/csr.hpp"

#include <stdexcept>
#include <utility>

namespace markov_cero::gpu {

DeviceCsr::DeviceCsr(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      row_offsets_(rows + 1), col_indices_(nnz), values_(nnz) {}

DeviceCsr DeviceCsr::from_csc(const linalg::SparseCsc& csc) {
    const std::size_t m = csc.rows;
    const std::size_t n = csc.columns;
    const std::size_t nnz = csc.values.size();

    if (csc.column_offsets.size() != n + 1) {
        throw std::invalid_argument("DeviceCsr::from_csc invalid column_offsets size");
    }
    if (csc.row_indices.size() != nnz) {
        throw std::invalid_argument("DeviceCsr::from_csc row_indices size mismatch");
    }

    std::vector<std::size_t> row_counts(m, 0);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = csc.column_offsets[j];
        const std::size_t end = csc.column_offsets[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            const std::size_t r = csc.row_indices[p];
            if (r >= m) {
                throw std::invalid_argument("DeviceCsr::from_csc row index out of range");
            }
            row_counts[r]++;
        }
    }

    std::vector<std::size_t> h_row_offsets(m + 1, 0);
    for (std::size_t i = 0; i < m; ++i) {
        h_row_offsets[i + 1] = h_row_offsets[i] + row_counts[i];
    }

    std::vector<std::size_t> h_col_indices(nnz, 0);
    std::vector<double> h_values(nnz, 0.0);
    std::vector<std::size_t> current_ptr = h_row_offsets;

    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = csc.column_offsets[j];
        const std::size_t end = csc.column_offsets[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            const std::size_t r = csc.row_indices[p];
            const std::size_t dest = current_ptr[r]++;
            h_col_indices[dest] = j;
            h_values[dest] = csc.values[p];
        }
    }

    DeviceCsr result(m, n, nnz);
    result.row_offsets_.upload(h_row_offsets);
    result.col_indices_.upload(h_col_indices);
    result.values_.upload(h_values);
    return result;
}

DeviceCsr DeviceCsr::from_csc(const model::SparseMatrixCSC& csc) {
    const std::size_t m = csc.row_count;
    const std::size_t n = csc.column_count;
    const std::size_t nnz = csc.value.size();

    if (csc.column_start.size() != n + 1) {
        throw std::invalid_argument("DeviceCsr::from_csc invalid column_start size");
    }
    if (csc.row_index.size() != nnz) {
        throw std::invalid_argument("DeviceCsr::from_csc row_index size mismatch");
    }

    std::vector<std::size_t> row_counts(m, 0);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = csc.column_start[j];
        const std::size_t end = csc.column_start[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            const std::size_t r = csc.row_index[p];
            if (r >= m) {
                throw std::invalid_argument("DeviceCsr::from_csc row index out of range");
            }
            row_counts[r]++;
        }
    }

    std::vector<std::size_t> h_row_offsets(m + 1, 0);
    for (std::size_t i = 0; i < m; ++i) {
        h_row_offsets[i + 1] = h_row_offsets[i] + row_counts[i];
    }

    std::vector<std::size_t> h_col_indices(nnz, 0);
    std::vector<double> h_values(nnz, 0.0);
    std::vector<std::size_t> current_ptr = h_row_offsets;

    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t start = csc.column_start[j];
        const std::size_t end = csc.column_start[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            const std::size_t r = csc.row_index[p];
            const std::size_t dest = current_ptr[r]++;
            h_col_indices[dest] = j;
            h_values[dest] = csc.value[p];
        }
    }

    DeviceCsr result(m, n, nnz);
    result.row_offsets_.upload(h_row_offsets);
    result.col_indices_.upload(h_col_indices);
    result.values_.upload(h_values);
    return result;
}

DeviceCsr DeviceCsr::transpose_from_csc(const linalg::SparseCsc& csc) {
    const std::size_t m = csc.rows;
    const std::size_t n = csc.columns;
    const std::size_t nnz = csc.values.size();

    if (csc.column_offsets.size() != n + 1) {
        throw std::invalid_argument("DeviceCsr::transpose_from_csc invalid column_offsets size");
    }
    if (csc.row_indices.size() != nnz) {
        throw std::invalid_argument("DeviceCsr::transpose_from_csc row_indices size mismatch");
    }

    DeviceCsr result(n, m, nnz);
    result.row_offsets_.upload(csc.column_offsets);
    result.col_indices_.upload(csc.row_indices);
    result.values_.upload(csc.values);
    return result;
}

DeviceCsr DeviceCsr::transpose_from_csc(const model::SparseMatrixCSC& csc) {
    const std::size_t m = csc.row_count;
    const std::size_t n = csc.column_count;
    const std::size_t nnz = csc.value.size();

    if (csc.column_start.size() != n + 1) {
        throw std::invalid_argument("DeviceCsr::transpose_from_csc invalid column_start size");
    }
    if (csc.row_index.size() != nnz) {
        throw std::invalid_argument("DeviceCsr::transpose_from_csc row_index size mismatch");
    }

    DeviceCsr result(n, m, nnz);
    result.row_offsets_.upload(csc.column_start);
    result.col_indices_.upload(csc.row_index);
    result.values_.upload(csc.value);
    return result;
}

void DeviceCsr::download_to(std::vector<std::size_t>& row_offsets,
                            std::vector<std::size_t>& col_indices,
                            std::vector<double>& values) const {
    row_offsets_.download(row_offsets);
    col_indices_.download(col_indices);
    values_.download(values);
}

linalg::SparseCsc DeviceCsr::to_csc_host() const {
    std::vector<std::size_t> h_row_offsets;
    std::vector<std::size_t> h_col_indices;
    std::vector<double> h_values;
    download_to(h_row_offsets, h_col_indices, h_values);

    std::vector<std::size_t> col_counts(cols_, 0);
    for (std::size_t i = 0; i < rows_; ++i) {
        const std::size_t start = h_row_offsets[i];
        const std::size_t end = h_row_offsets[i + 1];
        for (std::size_t p = start; p < end; ++p) {
            col_counts[h_col_indices[p]]++;
        }
    }

    std::vector<std::size_t> csc_col_offsets(cols_ + 1, 0);
    for (std::size_t j = 0; j < cols_; ++j) {
        csc_col_offsets[j + 1] = csc_col_offsets[j] + col_counts[j];
    }

    std::vector<std::size_t> csc_row_indices(nnz_, 0);
    std::vector<double> csc_values(nnz_, 0.0);
    std::vector<std::size_t> current_ptr = csc_col_offsets;

    for (std::size_t i = 0; i < rows_; ++i) {
        const std::size_t start = h_row_offsets[i];
        const std::size_t end = h_row_offsets[i + 1];
        for (std::size_t p = start; p < end; ++p) {
            const std::size_t c = h_col_indices[p];
            const std::size_t dest = current_ptr[c]++;
            csc_row_indices[dest] = i;
            csc_values[dest] = h_values[p];
        }
    }

    return linalg::SparseCsc{
        rows_, cols_,
        std::move(csc_col_offsets),
        std::move(csc_row_indices),
        std::move(csc_values)
    };
}

model::SparseMatrixCSC DeviceCsr::to_model_csc_host() const {
    linalg::SparseCsc csc = to_csc_host();
    return model::SparseMatrixCSC{
        csc.rows, csc.columns,
        std::move(csc.column_offsets),
        std::move(csc.row_indices),
        std::move(csc.values)
    };
}

} // namespace markov_cero::gpu
