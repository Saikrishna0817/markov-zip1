#include "markov_cero/gpu/buffer.hpp"
#include "markov_cero/gpu/csr.hpp"
#include "markov_cero/gpu/pdhg_step.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/model/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

markov_cero::model::Model load_mps(const std::string& filepath) {
    std::ifstream file(filepath);
    if (file.is_open()) {
        return markov_cero::io::parse_mps(file);
    }
    std::ifstream alt_file("../" + filepath);
    if (alt_file.is_open()) {
        return markov_cero::io::parse_mps(alt_file);
    }
    const char* src_dir = std::getenv("MARKOV_CERO_SOURCE_DIR");
    if (src_dir) {
        std::ifstream env_file(std::string(src_dir) + "/" + filepath);
        if (env_file.is_open()) {
            return markov_cero::io::parse_mps(env_file);
        }
    }
    throw std::runtime_error("Cannot open MPS file: " + filepath);
}

double compute_max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    require(a.size() == b.size(), "vector sizes must match");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

void test_synthetic_single_step() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    // Minimize -x0 - 2*x1
    // subject to:
    //   x0 + x1 <= 4
    //   x0 <= 2
    //   x0, x1 >= 0
    model::Model mdl;
    mdl.name = "synthetic_2x2";
    mdl.objective_sense = model::ObjectiveSense::minimize;
    mdl.objective = {-1.0, -2.0};
    mdl.objective_offset = 0.0;
    mdl.variable_name = {"x0", "x1"};
    mdl.variable_lower = {
        model::Bound{model::BoundKind::finite, 0.0},
        model::Bound{model::BoundKind::finite, 0.0}
    };
    mdl.variable_upper = {
        model::Bound{model::BoundKind::finite, 2.0},
        model::Bound{model::BoundKind::positive_infinity, 0.0}
    };
    mdl.row_name = {"c0", "c1"};
    mdl.row_lower = {
        model::Bound{model::BoundKind::negative_infinity, 0.0},
        model::Bound{model::BoundKind::negative_infinity, 0.0}
    };
    mdl.row_upper = {
        model::Bound{model::BoundKind::finite, 4.0},
        model::Bound{model::BoundKind::finite, 2.0}
    };

    model::SparseMatrixBuilder builder(2, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(1, 0, 1.0);
    mdl.matrix = builder.build();

    DeviceCsr A = DeviceCsr::from_csc(mdl.matrix);
    DeviceCsr At = DeviceCsr::transpose_from_csc(mdl.matrix);

    PdhgState state_gpu = create_pdhg_state(mdl, A, At, 0.9);
    PdhgState state_cpu = create_pdhg_state(mdl, A, At, 0.9);

    pdhg_step(state_gpu, 1);
    pdhg_step_cpu(state_cpu, 1);

    std::vector<double> x_gpu = state_gpu.x.to_vector();
    std::vector<double> x_cpu = state_cpu.x.to_vector();
    std::vector<double> y_gpu = state_gpu.y.to_vector();
    std::vector<double> y_cpu = state_cpu.y.to_vector();
    std::vector<double> xbar_gpu = state_gpu.x_bar.to_vector();
    std::vector<double> xbar_cpu = state_cpu.x_bar.to_vector();

    double diff_x = compute_max_abs_diff(x_gpu, x_cpu);
    double diff_y = compute_max_abs_diff(y_gpu, y_cpu);
    double diff_xbar = compute_max_abs_diff(xbar_gpu, xbar_cpu);

    require(diff_x <= 1e-12, "single step x diff exceeds 1e-12: " + std::to_string(diff_x));
    require(diff_y <= 1e-12, "single step y diff exceeds 1e-12: " + std::to_string(diff_y));
    require(diff_xbar <= 1e-12, "single step xbar diff exceeds 1e-12: " +
            std::to_string(diff_xbar));

    std::cout << "test_synthetic_single_step: PASS (diff_x="
              << std::scientific << std::setprecision(2) << diff_x << ")\n";
}

void test_netlib_multi_step(const std::string& name, const std::string& path) {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    model::Model mdl = load_mps(path);

    DeviceCsr A = DeviceCsr::from_csc(mdl.matrix);
    DeviceCsr At = DeviceCsr::transpose_from_csc(mdl.matrix);

    PdhgState state_gpu = create_pdhg_state(mdl, A, At, 0.9);
    PdhgState state_cpu = create_pdhg_state(mdl, A, At, 0.9);

    constexpr std::size_t num_steps = 50;
    pdhg_run_iterations(state_gpu, num_steps, 1);
    pdhg_run_iterations_cpu(state_cpu, num_steps, 1);

    std::vector<double> x_gpu = state_gpu.x.to_vector();
    std::vector<double> x_cpu = state_cpu.x.to_vector();
    std::vector<double> y_gpu = state_gpu.y.to_vector();
    std::vector<double> y_cpu = state_cpu.y.to_vector();
    std::vector<double> xavg_gpu = state_gpu.x_avg.to_vector();
    std::vector<double> xavg_cpu = state_cpu.x_avg.to_vector();
    std::vector<double> yavg_gpu = state_gpu.y_avg.to_vector();
    std::vector<double> yavg_cpu = state_cpu.y_avg.to_vector();

    double diff_x = compute_max_abs_diff(x_gpu, x_cpu);
    double diff_y = compute_max_abs_diff(y_gpu, y_cpu);
    double diff_xavg = compute_max_abs_diff(xavg_gpu, xavg_cpu);
    double diff_yavg = compute_max_abs_diff(yavg_gpu, yavg_cpu);

    require(diff_x <= 1e-12, name + " 50-step x diff exceeds 1e-12: " + std::to_string(diff_x));
    require(diff_y <= 1e-12, name + " 50-step y diff exceeds 1e-12: " + std::to_string(diff_y));
    require(diff_xavg <= 1e-12, name + " 50-step xavg diff exceeds 1e-12");
    require(diff_yavg <= 1e-12, name + " 50-step yavg diff exceeds 1e-12");

    // Invariant check: x_gpu must respect variable bounds
    std::vector<double> lo = state_gpu.var_lower.to_vector();
    std::vector<double> hi = state_gpu.var_upper.to_vector();
    for (std::size_t j = 0; j < mdl.matrix.column_count; ++j) {
        require(x_gpu[j] >= lo[j] - 1e-12, name + " x_gpu violates lower bound");
        require(x_gpu[j] <= hi[j] + 1e-12, name + " x_gpu violates upper bound");
    }

    std::cout << "test_netlib_multi_step (" << std::left << std::setw(10) << name
              << ", steps=" << num_steps << "): PASS (diff_x="
              << std::scientific << std::setprecision(2) << diff_x
              << " diff_y=" << diff_y << ")\n";
}

void test_zero_transfers_invariant() {
    using namespace markov_cero;
    using namespace markov_cero::gpu;

    model::Model mdl = load_mps("data/netlib/afiro.mps");

    DeviceCsr A = DeviceCsr::from_csc(mdl.matrix);
    DeviceCsr At = DeviceCsr::transpose_from_csc(mdl.matrix);
    PdhgState state = create_pdhg_state(mdl, A, At, 0.9);

    const double* orig_x_ptr = state.x.data();
    const double* orig_y_ptr = state.y.data();
    const double* orig_xbar_ptr = state.x_bar.data();
    const double* orig_xavg_ptr = state.x_avg.data();
    const double* orig_yavg_ptr = state.y_avg.data();
    const double* orig_Aty_ptr = state.At_y.data();
    const double* orig_Axbar_ptr = state.Ax_bar.data();

    pdhg_run_iterations(state, 100, 1);

    require(state.x.data() == orig_x_ptr, "x buffer reallocated unexpectedly");
    require(state.y.data() == orig_y_ptr, "y buffer reallocated unexpectedly");
    require(state.x_bar.data() == orig_xbar_ptr, "x_bar buffer reallocated unexpectedly");
    require(state.x_avg.data() == orig_xavg_ptr, "x_avg buffer reallocated unexpectedly");
    require(state.y_avg.data() == orig_yavg_ptr, "y_avg buffer reallocated unexpectedly");
    require(state.At_y.data() == orig_Aty_ptr, "At_y buffer reallocated unexpectedly");
    require(state.Ax_bar.data() == orig_Axbar_ptr, "Ax_bar buffer reallocated unexpectedly");

    std::cout << "test_zero_transfers_invariant: PASS (all device pointers stable)\n";
}

} // namespace

int main() {
    std::cout << "=== Markov-Cero Fused PDHG Step Tests (T-5.07) ===\n";
    test_synthetic_single_step();
    test_netlib_multi_step("afiro", "data/netlib/afiro.mps");
    test_netlib_multi_step("blend", "data/netlib/blend.mps");
    test_netlib_multi_step("adlittle", "data/netlib/adlittle.mps");
    test_zero_transfers_invariant();
    std::cout << "=== All Fused PDHG Step Tests Passed Successfully ===\n";
    return 0;
}
