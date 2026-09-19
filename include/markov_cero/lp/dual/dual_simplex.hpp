#pragma once

#include "markov_cero/lp/reference/revised_simplex.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace markov_cero::lp::dual {

// tableau_norm uses ||A^T B^{-T} e_i||^2. That is not conventional dual steepest-edge
// (||B^{-T} e_i||^2) and must not be advertised as exact DSE.
enum class PricingPolicy { bland, tableau_norm };

struct BasisState {
    std::size_t rows{};
    std::size_t columns{};
    std::string model_fingerprint;
    std::vector<std::size_t> basic_variables;
};

struct Options {
    std::size_t iteration_limit{100000};
    std::size_t telemetry_limit{10000};
    double feasibility_tolerance{1e-9};
    double dual_tolerance{1e-9};
    double pivot_tolerance{1e-12};
    double condition_trigger{1e-14};
    bool harris_ratio{true};
    bool allow_cold_fallback{true};
    PricingPolicy pricing{PricingPolicy::tableau_norm};
};

struct IterationRecord {
    std::size_t iteration{};
    double objective{};
    double most_negative_basic{};
    std::size_t leaving{};
    std::size_t entering{};
    double pivot{};
    bool harris{};
};

struct Result {
    reference::Result solution;
    BasisState basis_state;
    bool used_warm_start{};
    bool used_cold_fallback{};
    bool telemetry_truncated{};
    std::size_t refactorizations{};
    std::size_t perturbation_cleanups{};
    std::vector<IterationRecord> telemetry;
    std::string message;
};

[[nodiscard]] std::string fingerprint(const transform::CanonicalModel& model);
[[nodiscard]] BasisState make_basis_state(const transform::CanonicalModel& model,
                                          const std::vector<std::size_t>& basic_variables);
[[nodiscard]] std::string serialize_basis(const BasisState& basis);
[[nodiscard]] BasisState parse_basis(const std::string& text);
[[nodiscard]] Result solve(const transform::CanonicalModel& model, const Options& options = {},
                           const std::optional<BasisState>& warm_start = std::nullopt);

} // namespace markov_cero::lp::dual
