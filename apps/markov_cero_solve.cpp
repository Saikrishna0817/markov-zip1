#include "cli_options.hpp"
#include "json_output.hpp"
#include "markov_cero/foundation/build_info.hpp"
#include "markov_cero/io/mps.hpp"
#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/lp/first_order/pdlp.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/milp_solver.hpp"
#include "markov_cero/milp/parallel_tree_search.hpp"
#include "markov_cero/milp/strong_branching.hpp"
#include "markov_cero/presolve/presolve.hpp"
#include "markov_cero/scale/ruiz_scaling.hpp"
#include "markov_cero/transform/canonicalize.hpp"
#include "markov_cero/transform/sparse_canonical_model.hpp"
#include "markov_cero/verify/primal_verifier.hpp"
#include "markov_cero/verify/reference_lp_verifier.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


int main(int argc, char** argv) {
    using markov_cero::apps::json_number;
    auto cli = markov_cero::apps::CliOptions::parse(argc, argv);
    if (cli.help_requested) return 0;
    if (cli.error) return cli.exit_code;

    double pdlp_res_primal_infeas = 0.0;
    double pdlp_res_dual_infeas = 0.0;
    double pdlp_res_gap = 0.0;
    double pdlp_h2d_ms = 0.0;
    double pdlp_kernel_ms = 0.0;
    double pdlp_d2h_ms = 0.0;
    double pdlp_total_ms = 0.0;

const auto started = std::chrono::steady_clock::now();
    std::ifstream input(cli.path);
    if (!input) {
        std::cerr << "cannot open input\n";
        return 8;
    }

    bool original_verified = false;
    bool canonical_verified = false;
    std::string original_message;
    std::vector<double> original_primal;
    double original_objective = 0;
    markov_cero::verify::PrimalVerificationReport primal_report;
    markov_cero::verify::ReferenceVerification canonical_report;
    markov_cero::lp::reference::Result result;
    std::optional<markov_cero::lp::dual::BasisState> basis_to_save;
    bool used_warm_start = false;
    bool used_cold_fallback = false;
    markov_cero::presolve::PresolveResult presolve_res;
    markov_cero::scale::RuizScalers scalers;
    bool presolve_applied = false;
    bool scaling_applied = false;
    std::string error;

    std::string resolved_engine = cli.engine_name;
    std::size_t nodes_explored = 0;
    std::size_t total_lp_iterations = 0;
    double best_bound = 0.0;
    double relative_gap = 0.0;
    std::size_t cuts_generated = 0;
    std::size_t heuristics_found = 0;
    std::size_t model_rows = 0;
    std::size_t model_cols = 0;
    std::size_t model_nnz = 0;

    try {
        const auto model = markov_cero::io::parse_mps(input);
        model_rows = model.matrix.row_count;
        model_cols = model.matrix.column_count;
        model_nnz = model.matrix.value.size();

        bool has_discrete = false;
        for (const auto type : model.variable_type) {
            if (type != markov_cero::model::VariableType::continuous) {
                has_discrete = true;
                break;
            }
        }

        if (resolved_engine == "auto") {
            resolved_engine = has_discrete ? "milp" : "primal";
        }

        if (resolved_engine == "parallel") {
            markov_cero::milp::ParallelOptions par_opts;
            par_opts.num_threads = cli.num_threads;
            par_opts.time_limit_seconds = cli.milp_options.time_limit_seconds;
            par_opts.max_nodes = cli.milp_options.max_nodes;
            par_opts.enable_cuts = cli.milp_options.enable_cuts;
            par_opts.enable_mir_cuts = cli.milp_options.enable_mir_cuts;
            par_opts.enable_heuristics = cli.milp_options.enable_heuristics;
            par_opts.enable_strong_branching = cli.milp_options.enable_strong_branching;
            par_opts.branching_strategy = cli.milp_options.branching_strategy;
            const auto par_res = markov_cero::milp::solve_parallel(model, par_opts);
            result.status = par_res.status;
            result.message = par_res.message;
            nodes_explored = par_res.nodes_explored;
            total_lp_iterations = par_res.lp_iterations;
            best_bound = par_res.best_bound;
            relative_gap = par_res.relative_gap;
            cuts_generated = par_res.cuts_generated;
            heuristics_found = par_res.heuristics_found;

            if (result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                original_primal = par_res.primal;
                original_objective = par_res.objective;
                result.primal = par_res.primal;
                result.objective = par_res.objective;

                markov_cero::verify::Candidate candidate{original_primal, original_objective};
                primal_report = markov_cero::verify::verify_primal(model, candidate);
                original_verified = primal_report.passed;
                canonical_verified = true;
                std::string viol_desc;
                if (!primal_report.violations.empty()) {
                    const auto& v = primal_report.violations[0];
                    viol_desc = v.category + " idx=" + std::to_string(v.index) +
                                " act=" + std::to_string(v.actual) +
                                " bnd=" + std::to_string(v.bound) +
                                " diff=" + std::to_string(v.magnitude) +
                                " allow=" + std::to_string(v.allowance);
                }
                original_message = original_verified ? "original primal verified"
                                                     : ("original primal rejected: " + viol_desc);
                if (!original_verified) {
                    result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                    result.message = "original-model verification failed: " + viol_desc;
                }
            } else if (result.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                       result.status == markov_cero::lp::reference::SolveStatus::unbounded) {
                canonical_verified = true;
                original_message = "original primal not applicable";
            } else {
                original_message = "original primal not applicable";
            }
        } else if (resolved_engine == "pdlp") {
            markov_cero::lp::first_order::PdlpOptions pdlp_opts;
            pdlp_opts.backend = (cli.backend_name == "gpu")
                                    ? markov_cero::lp::first_order::Backend::gpu
                                    : markov_cero::lp::first_order::Backend::cpu;
            pdlp_opts.max_iterations =
                (cli.options.iteration_limit != 10000 && cli.options.iteration_limit > 0)
                    ? cli.options.iteration_limit
                    : 100000;
            pdlp_opts.set_tolerance(cli.pdlp_tolerance);
            const auto pdlp_res = markov_cero::lp::first_order::solve_pdlp(model, pdlp_opts);
            total_lp_iterations = pdlp_res.iterations;
            pdlp_res_primal_infeas = pdlp_res.primal_infeasibility;
            pdlp_res_dual_infeas = pdlp_res.dual_infeasibility;
            pdlp_res_gap = pdlp_res.duality_gap;
            pdlp_h2d_ms = pdlp_res.h2d_ms;
            pdlp_kernel_ms = pdlp_res.kernel_ms;
            pdlp_d2h_ms = pdlp_res.d2h_ms;
            pdlp_total_ms = pdlp_res.total_ms;
            if (pdlp_res.status == markov_cero::lp::first_order::PdlpStatus::optimal) {
                result.status = markov_cero::lp::reference::SolveStatus::optimal;
                result.primal = pdlp_res.primal;
                result.dual = pdlp_res.dual;
                result.objective = pdlp_res.objective;
                result.message = pdlp_res.message;
                original_primal = pdlp_res.primal;
                original_objective = pdlp_res.objective;

                markov_cero::verify::Candidate candidate{original_primal, original_objective};
                const markov_cero::verify::Tolerance pdlp_tol{
                    cli.pdlp_tolerance, cli.pdlp_tolerance};
                primal_report =
                    markov_cero::verify::verify_primal(model, candidate, pdlp_tol, pdlp_tol,
                                                       cli.pdlp_tolerance);
                original_verified = primal_report.passed;
                canonical_verified = true;
                std::string viol_desc;
                if (!primal_report.violations.empty()) {
                    const auto& v = primal_report.violations[0];
                    viol_desc = v.category + " idx=" + std::to_string(v.index) +
                                " act=" + std::to_string(v.actual) +
                                " bnd=" + std::to_string(v.bound) +
                                " diff=" + std::to_string(v.magnitude) +
                                " allow=" + std::to_string(v.allowance);
                }
                original_message = original_verified ? "original primal verified"
                                                     : ("original primal rejected: " + viol_desc);
                if (!original_verified) {
                    result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                    result.message = "original-model verification failed: " + viol_desc;
                }
            } else if (pdlp_res.status ==
                       markov_cero::lp::first_order::PdlpStatus::iteration_limit) {
                result.status = markov_cero::lp::reference::SolveStatus::iteration_limit;
                result.message = pdlp_res.message;
            } else {
                result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                result.message = pdlp_res.message;
            }
            nodes_explored = 1;
            best_bound = original_objective;
            relative_gap = 0.0;
        } else if (resolved_engine == "milp") {
            const auto milp_res = markov_cero::milp::solve(model, cli.milp_options);
            result.status = milp_res.status;
            result.message = milp_res.message;
            nodes_explored = milp_res.nodes_explored;
            total_lp_iterations = milp_res.lp_iterations;
            best_bound = milp_res.best_bound;
            relative_gap = milp_res.relative_gap;
            cuts_generated = milp_res.cuts_generated;
            heuristics_found = milp_res.heuristics_found;

            if (result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                original_primal = milp_res.primal;
                original_objective = milp_res.objective;
                result.primal = milp_res.primal;
                result.objective = milp_res.objective;

                markov_cero::verify::Candidate candidate{original_primal, original_objective};
                primal_report = markov_cero::verify::verify_primal(model, candidate);
                original_verified = primal_report.passed;
                canonical_verified = true;
                std::string viol_desc;
                if (!primal_report.violations.empty()) {
                    const auto& v = primal_report.violations[0];
                    viol_desc = v.category + " idx=" + std::to_string(v.index) +
                                " act=" + std::to_string(v.actual) +
                                " bnd=" + std::to_string(v.bound) +
                                " diff=" + std::to_string(v.magnitude) +
                                " allow=" + std::to_string(v.allowance);
                }
                original_message = original_verified ? "original primal verified"
                                                     : ("original primal rejected: " + viol_desc);
                if (!original_verified) {
                    result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                    result.message = "original-model verification failed: " + viol_desc;
                }
            } else if (result.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                       result.status == markov_cero::lp::reference::SolveStatus::unbounded) {
                canonical_verified = true;
                original_message = "original primal not applicable";
            } else {
                original_message = "original primal not applicable";
            }
        } else {
            const auto sparse_canonical =
                markov_cero::transform::sparse_canonicalize(model, /*relax_integrality=*/true);
            auto working_model = sparse_canonical;

            if (cli.enable_presolve) {
                markov_cero::presolve::PresolveOptions popts;
                popts.max_passes = cli.max_presolve_passes;
                presolve_res = markov_cero::presolve::presolve(sparse_canonical, popts);
                if (presolve_res.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                    presolve_res.status == markov_cero::lp::reference::SolveStatus::unbounded) {
                    result.status = presolve_res.status;
                    result.message = presolve_res.message;
                } else {
                    working_model = presolve_res.model;
                    presolve_applied = true;
                }
            }

            if (result.status != markov_cero::lp::reference::SolveStatus::infeasible &&
                result.status != markov_cero::lp::reference::SolveStatus::unbounded &&
                cli.enable_scale && working_model.matrix.rows > 0 &&
                working_model.matrix.columns > 0) {
                markov_cero::scale::RuizOptions ropts;
                ropts.max_iterations = cli.ruiz_iterations;
                scalers = markov_cero::scale::equilibrate(working_model, ropts);
                scaling_applied = true;
            }

            if (result.status != markov_cero::lp::reference::SolveStatus::infeasible &&
                result.status != markov_cero::lp::reference::SolveStatus::unbounded) {
                if (working_model.matrix.rows == 0 || working_model.matrix.columns == 0) {
                    result.status = markov_cero::lp::reference::SolveStatus::optimal;
                    result.primal.assign(working_model.matrix.columns, 0.0);
                    result.dual.assign(working_model.matrix.rows, 0.0);
                    result.objective = 0.0;
                } else {
                    const auto canonical = working_model.to_dense();
                    if (resolved_engine == "dual") {
                        markov_cero::lp::dual::Options dual_opts;
                        dual_opts.iteration_limit = cli.options.iteration_limit;
                        std::optional<markov_cero::lp::dual::BasisState> warm_basis;
                        if (!cli.warm_start_path.empty()) {
                            std::ifstream bfile(cli.warm_start_path);
                            if (!bfile) {
                                throw std::invalid_argument("cannot open warm-start basis file");
                            }
                            std::string btext((std::istreambuf_iterator<char>(bfile)),
                                              std::istreambuf_iterator<char>());
                            warm_basis = markov_cero::lp::dual::parse_basis(btext);
                        }
                        const auto dual_res =
                            markov_cero::lp::dual::solve(canonical, dual_opts, warm_basis);
                        result = dual_res.solution;
                        basis_to_save = dual_res.basis_state;
                        used_warm_start = dual_res.used_warm_start;
                        used_cold_fallback = dual_res.used_cold_fallback;
                    } else {
                        result = markov_cero::lp::reference::solve(canonical, cli.options);
                        if (result.status == markov_cero::lp::reference::SolveStatus::optimal &&
                            result.basis.size() == canonical.matrix.rows) {
                            basis_to_save =
                                markov_cero::lp::dual::make_basis_state(canonical, result.basis);
                        }
                    }
                }
            }

            if (scaling_applied &&
                result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                markov_cero::scale::unscale_solution(scalers, result);
            }

            if (presolve_applied &&
                result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                result =
                    markov_cero::presolve::postsolve(presolve_res.stack, result, sparse_canonical);
            }

            if (result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                const auto Ax = sparse_canonical.multiply(result.primal);
                double max_viol = 0.0;
                for (std::size_t i = 0; i < sparse_canonical.rhs.size(); ++i) {
                    max_viol = std::max(max_viol, std::abs(Ax[i] - sparse_canonical.rhs[i]));
                }
                canonical_verified =
                    (max_viol <= std::max(cli.options.feasibility_tolerance,
                                          cli.options.dual_tolerance));

                if (result.dual.size() == sparse_canonical.matrix.rows) {
                    const auto aty = sparse_canonical.multiply_transpose(result.dual);
                    double max_dual_viol = 0.0;
                    for (std::size_t j = 0; j < sparse_canonical.objective.size(); ++j) {
                        const double rc = sparse_canonical.objective[j] - aty[j];
                        if (-rc > max_dual_viol) {
                            max_dual_viol = -rc;
                        }
                    }
                    canonical_report.maximum_dual_violation = max_dual_viol;
                    canonical_verified =
                        canonical_verified &&
                        (max_dual_viol <= std::max(cli.options.feasibility_tolerance,
                                                   cli.options.dual_tolerance));
                }
            } else if (result.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                       result.status == markov_cero::lp::reference::SolveStatus::unbounded) {
                canonical_verified = true;
            }

            if ((result.status == markov_cero::lp::reference::SolveStatus::optimal ||
                 result.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                 result.status == markov_cero::lp::reference::SolveStatus::unbounded) &&
                !canonical_verified) {
                result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                result.message = "canonical witness rejected: " + canonical_report.message;
            }

            if (result.status == markov_cero::lp::reference::SolveStatus::optimal) {
                original_primal =
                    markov_cero::transform::reconstruct_primal(sparse_canonical, result.primal);
                original_objective = markov_cero::transform::reconstruct_objective(
                    sparse_canonical, result.objective);
                markov_cero::verify::Candidate candidate{original_primal, original_objective};
                primal_report = markov_cero::verify::verify_primal(model, candidate);
                original_verified = primal_report.passed;
                original_message =
                    original_verified ? "original primal verified" : "original primal rejected";
                if (!original_verified) {
                    result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
                    result.message = "original-model verification failed";
                } else if (!cli.save_basis_path.empty() && basis_to_save.has_value()) {
                    std::ofstream bfile(cli.save_basis_path);
                    if (bfile) {
                        bfile << markov_cero::lp::dual::serialize_basis(*basis_to_save);
                    }
                }
                nodes_explored = 1;
                best_bound = original_objective;
                relative_gap = 0.0;
            } else {
                original_message = "original primal not applicable";
            }
            total_lp_iterations = result.phase_one_iterations + result.phase_two_iterations;
        }
    } catch (const markov_cero::io::MpsError& e) {
        result.status = markov_cero::lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        error = e.what();
    } catch (const std::invalid_argument& e) {
        result.status = markov_cero::lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        error = e.what();
    } catch (const std::length_error& e) {
        result.status = markov_cero::lp::reference::SolveStatus::resource_limit;
        result.message = e.what();
        error = e.what();
    } catch (const std::exception& e) {
        result.status = markov_cero::lp::reference::SolveStatus::numerical_failure;
        result.message = e.what();
        error = e.what();
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    const bool verified = (result.status == markov_cero::lp::reference::SolveStatus::optimal &&
                           original_verified && canonical_verified) ||
                          ((result.status == markov_cero::lp::reference::SolveStatus::infeasible ||
                            result.status == markov_cero::lp::reference::SolveStatus::unbounded) &&
                           canonical_verified);

        markov_cero::apps::JsonOutputData out_data;
    out_data.resolved_engine = resolved_engine;
    out_data.result = result;
    out_data.model_rows = model_rows;
    out_data.model_cols = model_cols;
    out_data.model_nnz = model_nnz;
    out_data.verified = verified;
    out_data.original_objective = original_objective;
    out_data.original_primal = original_primal;
    out_data.canonical_verified = canonical_verified;
    out_data.original_verified = original_verified;
    out_data.original_message = original_message;
    out_data.used_warm_start = used_warm_start;
    out_data.used_cold_fallback = used_cold_fallback;
    out_data.primal_report = primal_report;
    out_data.canonical_report = canonical_report;
    out_data.elapsed_ms = elapsed_ms;
    out_data.nodes_explored = nodes_explored;
    out_data.total_lp_iterations = total_lp_iterations;
    out_data.best_bound = best_bound;
    out_data.relative_gap = relative_gap;
    out_data.cuts_generated = cuts_generated;
    out_data.heuristics_found = heuristics_found;
    out_data.pdlp_tolerance = cli.pdlp_tolerance;
    out_data.pdlp_res_primal_infeas = pdlp_res_primal_infeas;
    out_data.pdlp_res_dual_infeas = pdlp_res_dual_infeas;
    out_data.pdlp_res_gap = pdlp_res_gap;
    out_data.backend_name = cli.backend_name;
    out_data.pdlp_h2d_ms = pdlp_h2d_ms;
    out_data.pdlp_kernel_ms = pdlp_kernel_ms;
    out_data.pdlp_d2h_ms = pdlp_d2h_ms;
    out_data.pdlp_total_ms = pdlp_total_ms;
    out_data.error = error;
    out_data.output_path = cli.output_path;
    markov_cero::apps::emit_json_output(out_data);

std::string timing_diag;
    if (resolved_engine == "pdlp") {
        if (cli.backend_name == "gpu") {
            timing_diag = " [gpu H2D=" + json_number(pdlp_h2d_ms) + "ms kernel=" +
                          json_number(pdlp_kernel_ms) + "ms D2H=" + json_number(pdlp_d2h_ms) +
                          "ms total=" + json_number(pdlp_total_ms) + "ms]";
        } else {
            timing_diag = " [cpu total=" + json_number(pdlp_total_ms) + "ms]";
        }
    }

    std::cerr << "markov-cero " << markov_cero::foundation::version() << " "
              << markov_cero::lp::reference::to_string(result.status)
              << (resolved_engine == "pdlp"
                      ? (" [tol=" + json_number(cli.pdlp_tolerance) + "]" + timing_diag)
                      : "")
              << (verified ? " VERIFIED\n" : " NOT VERIFIED\n");
    return markov_cero::apps::exit_code(result.status);
}
