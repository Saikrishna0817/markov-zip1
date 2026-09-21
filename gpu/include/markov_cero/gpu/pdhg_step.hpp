#pragma once

#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"
#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero {

namespace scale {
struct RuizScalers;
}

namespace gpu {

// Device-resident state for First-Order Primal-Dual Hybrid Gradient (PDHG / PDLP)
struct PdhgState {
    const DeviceCsr* A{nullptr};
    const DeviceCsr* At{nullptr};

    // Problem coefficients and diagonal preconditioners
    DeviceBuffer<double> c;
    DeviceBuffer<double> var_lower;
    DeviceBuffer<double> var_upper;
    DeviceBuffer<double> row_lower;
    DeviceBuffer<double> row_upper;
    DeviceBuffer<double> tau;
    DeviceBuffer<double> sigma;

    // Primal and dual iterates (resident in device memory)
    DeviceBuffer<double> x;
    DeviceBuffer<double> x_bar;
    DeviceBuffer<double> y;
    DeviceBuffer<double> x_avg;
    DeviceBuffer<double> y_avg;

    // SpMV workspace buffers (resident in device memory)
    DeviceBuffer<double> At_y;
    DeviceBuffer<double> Ax_bar;

    std::size_t num_variables{0};
    std::size_t num_constraints{0};

    std::vector<double> col_norms;
    std::vector<double> row_norms;
    double eta{0.99};
    double omega{1.0};
};

// Initialize device-resident PDHG state from Model and pre-built CSR matrices
PdhgState create_pdhg_state(const model::Model& model,
                            const DeviceCsr& A,
                            const DeviceCsr& At,
                            double step_size_reduction = 0.9,
                            double primal_weight = 1.0);

// Update device-resident step sizes tau and sigma from current eta and omega
void pdhg_update_step_sizes(PdhgState& state, double eta, double omega);

// Execute a single fused PDHG iteration (dispatches to CUDA if enabled, else CPU)
void pdhg_step(PdhgState& state, std::size_t avg_count);

// CPU reference implementation of single fused PDHG iteration
void pdhg_step_cpu(PdhgState& state, std::size_t avg_count);

// Residuals and normalized duality gap metrics for PDHG convergence and restarts
struct PdhgResiduals {
    double primal_infeasibility{0.0};
    double dual_infeasibility{0.0};
    double duality_gap{0.0};
    double score{0.0}; // max(primal_infeasibility, dual_infeasibility, duality_gap)
};

// Evaluate primal/dual residuals and duality gap on candidate average (x_avg, y_avg)
PdhgResiduals evaluate_residuals(const PdhgState& state,
                                 const model::Model& original_model,
                                 const scale::RuizScalers* scalers = nullptr);

// Reset base iterates x, y, and x_bar to current candidate ergodic averages
void pdhg_restart(PdhgState& state);

// Run multiple PDHG iterations entirely on device with ZERO H2D/D2H memory transfers
void pdhg_run_iterations(PdhgState& state,
                         std::size_t num_iterations,
                         std::size_t start_avg_count = 1);

// CPU reference for running multiple iterations
void pdhg_run_iterations_cpu(PdhgState& state,
                             std::size_t num_iterations,
                             std::size_t start_avg_count = 1);

// Solve LP using matrix-free GPU PDLP with adaptive restart on normalized duality gap
markov_cero::lp::first_order::PdlpResult solve_pdlp_gpu(
    const model::Model& model,
    const markov_cero::lp::first_order::PdlpOptions& options = {});

namespace detail {

void launch_pdhg_primal_step(std::size_t n,
                             const double* tau,
                             const double* c,
                             const double* At_y,
                             const double* var_lower,
                             const double* var_upper,
                             double* x,
                             double* x_bar,
                             double* x_avg,
                             std::size_t avg_count);

void launch_pdhg_dual_step(std::size_t m,
                           const double* sigma,
                           const double* Ax_bar,
                           const double* row_lower,
                           const double* row_upper,
                           double* y,
                           double* y_avg,
                           std::size_t avg_count);

void pdhg_primal_step_cpu(std::size_t n,
                          const double* tau,
                          const double* c,
                          const double* At_y,
                          const double* var_lower,
                          const double* var_upper,
                          double* x,
                          double* x_bar,
                          double* x_avg,
                          std::size_t avg_count);

void pdhg_dual_step_cpu(std::size_t m,
                        const double* sigma,
                        const double* Ax_bar,
                        const double* row_lower,
                        const double* row_upper,
                        double* y,
                        double* y_avg,
                        std::size_t avg_count);

} // namespace detail

} // namespace gpu

} // namespace markov_cero
