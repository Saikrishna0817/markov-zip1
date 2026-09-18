#pragma once
#include "sihopt/transform/canonicalize.hpp"
#include <cstddef>
#include <string>
#include <vector>
namespace sihopt::lp::reference {
enum class SolveStatus { optimal, infeasible, unbounded, iteration_limit, invalid_model, invalid_options, resource_limit, numerical_failure };
struct Options { std::size_t iteration_limit{10000}; std::size_t telemetry_limit{10000}; double feasibility_tolerance{1e-9}; double dual_tolerance{1e-9}; double pivot_tolerance{1e-12}; bool bland_anti_cycling{true}; };
struct IterationRecord { std::size_t iteration{}; int phase{}; double objective{}; double minimum_reduced_cost{}; std::size_t entering{}; std::size_t leaving{}; bool degenerate{}; };
struct Result { SolveStatus status{SolveStatus::numerical_failure}; std::vector<double> primal; std::vector<double> dual; std::vector<double> ray; std::vector<double> certificate; std::vector<std::size_t> basis; double objective{}; std::size_t phase_one_iterations{}; std::size_t phase_two_iterations{}; std::size_t bound_flips{}; bool telemetry_truncated{}; std::vector<IterationRecord> telemetry; std::string message; };
[[nodiscard]] Result solve(const transform::CanonicalModel& model,const Options& options={});
[[nodiscard]] const char* to_string(SolveStatus status) noexcept;
}
