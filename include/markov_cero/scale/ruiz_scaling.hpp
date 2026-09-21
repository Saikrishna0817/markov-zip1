#pragma once

#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/model/model.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cstddef>
#include <vector>

namespace markov_cero::scale {

struct RuizOptions {
    std::size_t max_iterations{10};
    double tolerance{1e-3};
    double min_scale{1e-4};
    double max_scale{1e4};
};

struct RuizScalers {
    std::vector<double> row_scale;     // D_R
    std::vector<double> col_scale;     // D_C
    std::vector<double> inv_row_scale; // D_R^{-1}
    std::vector<double> inv_col_scale; // D_C^{-1}
    std::size_t iterations_executed{};
    bool converged{false};
};

[[nodiscard]] RuizScalers equilibrate(transform::SparseCanonicalModel& model,
                                      const RuizOptions& options = {});

void unscale_solution(const RuizScalers& scalers, lp::reference::Result& solution);

[[nodiscard]] RuizScalers equilibrate_model(model::Model& model,
                                            const RuizOptions& options = {});

void unscale_model_solution(const RuizScalers& scalers,
                            std::vector<double>& primal,
                            std::vector<double>& dual);

} // namespace markov_cero::scale
