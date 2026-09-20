#pragma once

#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/presolve/presolve_stack.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

#include <cstddef>
#include <string>

namespace markov_cero::presolve {

struct PresolveOptions {
    std::size_t max_passes{5};
    double feasibility_tolerance{1e-9};
    double dual_tolerance{1e-9};
    double pivot_tolerance{1e-12};
};

struct PresolveStatistics {
    std::size_t passes_executed{};
    std::size_t empty_rows_removed{};
    std::size_t empty_cols_removed{};
    std::size_t fixed_vars_removed{};
    std::size_t row_singletons_removed{};
    std::size_t original_rows{};
    std::size_t original_cols{};
    std::size_t presolved_rows{};
    std::size_t presolved_cols{};
};

struct PresolveResult {
    lp::reference::SolveStatus status{lp::reference::SolveStatus::optimal};
    transform::SparseCanonicalModel model;
    PresolveStack stack;
    PresolveStatistics statistics;
    std::string message;
};

[[nodiscard]] PresolveResult presolve(const transform::SparseCanonicalModel& input,
                                      const PresolveOptions& options = {});

[[nodiscard]] lp::reference::Result postsolve(const PresolveStack& stack,
                                              const lp::reference::Result& reduced_solution,
                                              const transform::SparseCanonicalModel& original_model,
                                              double tolerance = 1e-8);

} // namespace markov_cero::presolve
