#include "markov_cero/milp/node_lp.hpp"

#include "markov_cero/qp/admm_solver.hpp"
#include "markov_cero/qp/model.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"

namespace markov_cero::milp {

NodeLpResult solve_node_relaxation(
    const model::Model& node_model,
    const Options& options,
    const std::optional<lp::dual::BasisState>& warm_start) {
    NodeLpResult res;

    if (node_model.has_quadratic_objective) {
        try {
            const auto qp = qp::make_quadratic_model(node_model);
            qp::QpOptions qopts;
            qopts.max_iterations = options.max_iterations;
            qopts.absolute_tolerance = options.feasibility_tolerance;
            qopts.relative_tolerance = options.feasibility_tolerance;
            const auto qpres = qp::solve_qp(qp, qopts);

            if (qpres.status == qp::QpStatus::optimal) {
                res.status = lp::reference::SolveStatus::optimal;
                res.primal = qpres.x;
                res.objective = qpres.objective_value;
                res.iterations = qpres.iterations;
            } else if (qpres.status == qp::QpStatus::primal_infeasible) {
                res.status = lp::reference::SolveStatus::infeasible;
            } else {
                res.status = lp::reference::SolveStatus::numerical_failure;
            }
        } catch (...) {
            res.status = lp::reference::SolveStatus::numerical_failure;
        }
        return res;
    }

    try {
        const auto canon =
            transform::sparse_canonicalize(node_model, /*relax_integrality=*/true);
        const auto dense = canon.to_dense();

        if (options.enable_warm_start && warm_start.has_value()) {
            lp::dual::Options dopts;
            dopts.iteration_limit = options.max_iterations;
            dopts.feasibility_tolerance = options.feasibility_tolerance;
            dopts.allow_cold_fallback = true;
            const auto dres = lp::dual::solve(dense, dopts, warm_start);
            res.status = dres.solution.status;
            res.iterations =
                dres.solution.phase_one_iterations + dres.solution.phase_two_iterations;
            if (res.status == lp::reference::SolveStatus::optimal) {
                res.primal = transform::reconstruct_primal(canon, dres.solution.primal);
                res.objective =
                    transform::reconstruct_objective(canon, dres.solution.objective);
                res.basis = dres.basis_state;
            }
        } else {
            lp::reference::Options ropts;
            ropts.iteration_limit = options.max_iterations;
            ropts.feasibility_tolerance = options.feasibility_tolerance;
            const auto rres = lp::reference::solve(dense, ropts);
            res.status = rres.status;
            res.iterations = rres.phase_one_iterations + rres.phase_two_iterations;
            if (res.status == lp::reference::SolveStatus::optimal) {
                res.primal = transform::reconstruct_primal(canon, rres.primal);
                res.objective = transform::reconstruct_objective(canon, rres.objective);
                if (rres.basis.size() == dense.matrix.rows) {
                    try {
                        res.basis = lp::dual::make_basis_state(dense, rres.basis);
                    } catch (...) {
                    }
                }
            }
        }
    } catch (...) {
        res.status = lp::reference::SolveStatus::numerical_failure;
    }
    return res;
}

} // namespace markov_cero::milp
