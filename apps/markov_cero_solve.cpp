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

namespace {

int exit_code(markov_cero::lp::reference::SolveStatus status) {
    using markov_cero::lp::reference::SolveStatus;
    switch (status) {
    case SolveStatus::optimal:
        return 0;
    case SolveStatus::infeasible:
        return 1;
    case SolveStatus::unbounded:
        return 2;
    case SolveStatus::invalid_model:
        return 3;
    case SolveStatus::invalid_options:
        return 4;
    case SolveStatus::resource_limit:
        return 5;
    case SolveStatus::iteration_limit:
        return 6;
    case SolveStatus::numerical_failure:
        return 7;
    }
    return 7;
}

std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(c));
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
            out += buf;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream o;
    o.setf(std::ios::fmtflags(0), std::ios::floatfield);
    o.precision(17);
    o << value;
    return o.str();
}

std::string json_array(const std::vector<double>& values) {
    std::ostringstream o;
    o << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            o << ',';
        }
        o << json_number(values[i]);
    }
    o << ']';
    return o.str();
}

void usage(std::ostream& out) {
    out << "usage: markov-cero-solve MODEL.mps [options]\n"
        << "options:\n"
        << "  --output result.json     Write output JSON to file\n"
        << "  --engine primal|dual|pdlp|milp|parallel|auto Select solver engine (default: auto)\n"
        << "  --threads N              Worker threads for parallel tree search (default: 4)\n"
        << "  --branching most_fractional|pseudo_cost|strong_branching|reliability Branching "
           "variable selection rule (default: pseudo_cost)\n"
        << "  --iteration-limit N      Maximum simplex iterations\n"
        << "  --max-nodes N            Maximum branch-and-cut search nodes (default: 50000)\n"
        << "  --time-limit SEC         Maximum search time limit in seconds (default: 60.0)\n"
        << "  --cuts, --no-cuts        Enable or disable Gomory & MIR mixed-integer cuts (default: "
           "enabled)\n"
        << "  --heuristics, --no-heuristics Enable or disable primal heuristics (default: "
           "enabled)\n"
        << "  --warm-start FILE        Load warm-start basis from file (dual engine)\n"
        << "  --save-basis FILE        Save optimal basis to file\n"
        << "  --presolve, --no-presolve Enable or disable presolve reductions (default: enabled)\n"
        << "  --scale, --no-scale       Enable or disable Ruiz matrix scaling (default: enabled)\n"
        << "  --max-presolve-passes N   Maximum presolve passes (default: 5)\n"
        << "  --ruiz-iterations N       Maximum Ruiz equilibration iterations (default: 10)\n"
        << "  --tolerance TOL          Relative KKT tolerance for PDLP (default: 1e-4)\n"
        << "  --backend cpu|gpu        PDLP execution backend (default: cpu)\n"
        << "  --help, -h               Show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    std::string output_path;
    std::string engine_name = "auto";
    std::size_t num_threads = 4;
    std::string warm_start_path;
    std::string save_basis_path;
    bool enable_presolve = true;
    bool enable_scale = true;
    std::size_t max_presolve_passes = 5;
    std::size_t ruiz_iterations = 10;
    double pdlp_tolerance = 1e-4;
    std::string backend_name = "cpu";
    double pdlp_res_primal_infeas = 0.0;
    double pdlp_res_dual_infeas = 0.0;
    double pdlp_res_gap = 0.0;
    double pdlp_h2d_ms = 0.0;
    double pdlp_kernel_ms = 0.0;
    double pdlp_d2h_ms = 0.0;
    double pdlp_total_ms = 0.0;
    markov_cero::lp::reference::Options options;
    markov_cero::milp::Options milp_options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            return 0;
        }
        if (arg == "--output") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            output_path = argv[++i];
            continue;
        }
        if (arg == "--engine") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            engine_name = argv[++i];
            if (engine_name != "primal" && engine_name != "dual" && engine_name != "pdlp" &&
                engine_name != "milp" && engine_name != "parallel" && engine_name != "auto") {
                std::cerr
                    << "invalid engine (must be primal, dual, pdlp, milp, parallel, or auto): "
                    << engine_name << "\n";
                return 8;
            }
            continue;
        }
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            num_threads = std::strtoul(argv[++i], nullptr, 10);
            if (num_threads == 0) {
                num_threads = 1;
            }
            continue;
        }
        if (arg == "--branching") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            const std::string bval = argv[++i];
            if (bval == "most_fractional") {
                milp_options.branching_strategy =
                    markov_cero::milp::BranchingStrategy::most_fractional;
            } else if (bval == "pseudo_cost") {
                milp_options.branching_strategy = markov_cero::milp::BranchingStrategy::pseudo_cost;
            } else if (bval == "strong_branching") {
                milp_options.branching_strategy =
                    markov_cero::milp::BranchingStrategy::strong_branching;
            } else if (bval == "reliability") {
                milp_options.branching_strategy = markov_cero::milp::BranchingStrategy::reliability;
            } else {
                std::cerr << "invalid branching strategy (most_fractional, pseudo_cost, "
                             "strong_branching, reliability): "
                          << bval << "\n";
                return 8;
            }
            continue;
        }
        if (arg == "--max-nodes") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            milp_options.max_nodes = std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--time-limit") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            milp_options.time_limit_seconds = std::strtod(argv[++i], nullptr);
            continue;
        }
        if (arg == "--cuts") {
            milp_options.enable_cuts = true;
            continue;
        }
        if (arg == "--no-cuts") {
            milp_options.enable_cuts = false;
            continue;
        }
        if (arg == "--heuristics") {
            milp_options.enable_heuristics = true;
            continue;
        }
        if (arg == "--no-heuristics") {
            milp_options.enable_heuristics = false;
            continue;
        }
        if (arg == "--warm-start") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            warm_start_path = argv[++i];
            continue;
        }
        if (arg == "--save-basis") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            save_basis_path = argv[++i];
            continue;
        }
        if (arg == "--presolve") {
            enable_presolve = true;
            continue;
        }
        if (arg == "--no-presolve") {
            enable_presolve = false;
            continue;
        }
        if (arg == "--scale") {
            enable_scale = true;
            continue;
        }
        if (arg == "--no-scale") {
            enable_scale = false;
            continue;
        }
        if (arg == "--max-presolve-passes") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            max_presolve_passes = std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--ruiz-iterations") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            ruiz_iterations = std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--tolerance") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            pdlp_tolerance = std::strtod(argv[++i], nullptr);
            if (pdlp_tolerance <= 0.0) {
                std::cerr << "tolerance must be positive: " << pdlp_tolerance << "\n";
                return 8;
            }
            continue;
        }
        if (arg == "--backend") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            backend_name = argv[++i];
            if (backend_name != "cpu" && backend_name != "gpu") {
                std::cerr << "invalid backend (must be cpu or gpu): " << backend_name << "\n";
                return 8;
            }
            continue;
        }
        if (arg == "--iteration-limit") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            char* endptr = nullptr;
            errno = 0;
            const unsigned long long val = std::strtoull(argv[++i], &endptr, 10);
            if (errno == ERANGE || endptr == argv[i] || *endptr != '\0') {
                std::cerr << "invalid iteration limit: " << argv[i] << "\n";
                return 8;
            }
            options.iteration_limit = static_cast<std::size_t>(val);
            milp_options.max_iterations = static_cast<std::size_t>(val);
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            usage(std::cerr);
            return 8;
        }
        if (!path.empty()) {
            usage(std::cerr);
            return 8;
        }
        path = arg;
    }
    if (path.empty()) {
        usage(std::cerr);
        return 8;
    }

    const auto started = std::chrono::steady_clock::now();
    std::ifstream input(path);
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

    std::string resolved_engine = engine_name;
    std::size_t nodes_explored = 0;
    std::size_t total_lp_iterations = 0;
    double best_bound = 0.0;
    double relative_gap = 0.0;
    std::size_t cuts_generated = 0;
    std::size_t heuristics_found = 0;

    try {
        const auto model = markov_cero::io::parse_mps(input);

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
            par_opts.num_threads = num_threads;
            par_opts.time_limit_seconds = milp_options.time_limit_seconds;
            par_opts.max_nodes = milp_options.max_nodes;
            par_opts.enable_cuts = milp_options.enable_cuts;
            par_opts.enable_mir_cuts = milp_options.enable_mir_cuts;
            par_opts.enable_heuristics = milp_options.enable_heuristics;
            par_opts.enable_strong_branching = milp_options.enable_strong_branching;
            par_opts.branching_strategy = milp_options.branching_strategy;
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
            pdlp_opts.backend = (backend_name == "gpu")
                                    ? markov_cero::lp::first_order::Backend::gpu
                                    : markov_cero::lp::first_order::Backend::cpu;
            pdlp_opts.max_iterations =
                (options.iteration_limit != 10000 && options.iteration_limit > 0)
                    ? options.iteration_limit
                    : 100000;
            pdlp_opts.set_tolerance(pdlp_tolerance);
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
                const markov_cero::verify::Tolerance pdlp_tol{pdlp_tolerance, pdlp_tolerance};
                primal_report =
                    markov_cero::verify::verify_primal(model, candidate, pdlp_tol, pdlp_tol,
                                                       pdlp_tolerance);
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
            const auto milp_res = markov_cero::milp::solve(model, milp_options);
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

            if (enable_presolve) {
                markov_cero::presolve::PresolveOptions popts;
                popts.max_passes = max_presolve_passes;
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
                enable_scale && working_model.matrix.rows > 0 && working_model.matrix.columns > 0) {
                markov_cero::scale::RuizOptions ropts;
                ropts.max_iterations = ruiz_iterations;
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
                        dual_opts.iteration_limit = options.iteration_limit;
                        std::optional<markov_cero::lp::dual::BasisState> warm_basis;
                        if (!warm_start_path.empty()) {
                            std::ifstream bfile(warm_start_path);
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
                        result = markov_cero::lp::reference::solve(canonical, options);
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
                canonical_report.maximum_primal_violation = max_viol;
                canonical_verified =
                    (max_viol <= std::max(options.feasibility_tolerance, options.dual_tolerance));

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
                    canonical_verified = canonical_verified &&
                                         (max_dual_viol <= std::max(options.feasibility_tolerance,
                                                                    options.dual_tolerance));
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
                } else if (!save_basis_path.empty() && basis_to_save.has_value()) {
                    std::ofstream bfile(save_basis_path);
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

    std::ostringstream json;
    json << "{\"version\":\"" << json_escape(std::string(markov_cero::foundation::version()))
         << "\","
         << "\"milestone\":\"" << json_escape(std::string(markov_cero::foundation::milestone()))
         << "\","
         << "\"engine\":\"" << json_escape(resolved_engine) << "\","
         << "\"status\":\"" << markov_cero::lp::reference::to_string(result.status) << "\","
         << "\"verified\":" << (verified ? "true" : "false") << ","
         << "\"message\":\"" << json_escape(result.message) << "\","
         << "\"objective\":"
         << json_number(result.status == markov_cero::lp::reference::SolveStatus::optimal
                            ? original_objective
                            : result.objective)
         << ","
         << "\"primal\":" << json_array(original_primal.empty() ? result.primal : original_primal)
         << ","
         << "\"canonical_verified\":" << (canonical_verified ? "true" : "false") << ","
         << "\"original_verified\":" << (original_verified ? "true" : "false") << ","
         << "\"original_message\":\"" << json_escape(original_message) << "\","
         << "\"used_warm_start\":" << (used_warm_start ? "true" : "false") << ","
         << "\"used_cold_fallback\":" << (used_cold_fallback ? "true" : "false") << ","
         << "\"maximum_primal_violation\":" << json_number(primal_report.maximum_row_violation)
         << ","
         << "\"maximum_variable_violation\":"
         << json_number(primal_report.maximum_variable_violation) << ","
         << "\"maximum_integrality_violation\":"
         << json_number(primal_report.maximum_integrality_violation) << ","
         << "\"maximum_canonical_primal_violation\":"
         << json_number(canonical_report.maximum_primal_violation) << ","
         << "\"maximum_canonical_dual_violation\":"
         << json_number(canonical_report.maximum_dual_violation) << ","
         << "\"runtime_ms\":" << json_number(elapsed_ms) << ","
         << "\"nodes_explored\":" << nodes_explored << ","
         << "\"lp_iterations\":" << total_lp_iterations << ","
         << "\"best_bound\":" << json_number(best_bound) << ","
         << "\"relative_gap\":" << json_number(relative_gap) << ","
         << "\"cuts_generated\":" << cuts_generated << ","
         << "\"heuristics_found\":" << heuristics_found << ","
         << "\"phase_one_iterations\":" << result.phase_one_iterations << ","
         << "\"phase_two_iterations\":" << result.phase_two_iterations << ","
         << "\"pdlp_tolerance\":" << json_number(pdlp_tolerance) << ","
         << "\"relative_primal_residual\":" << json_number(pdlp_res_primal_infeas) << ","
         << "\"relative_dual_residual\":" << json_number(pdlp_res_dual_infeas) << ","
         << "\"relative_duality_gap\":" << json_number(pdlp_res_gap) << ","
         << "\"backend\":\"" << json_escape(backend_name) << "\","
         << "\"h2d_ms\":" << json_number(pdlp_h2d_ms) << ","
         << "\"kernel_ms\":" << json_number(pdlp_kernel_ms) << ","
         << "\"d2h_ms\":" << json_number(pdlp_d2h_ms) << ","
         << "\"total_ms\":"
         << json_number(resolved_engine == "pdlp" ? pdlp_total_ms : elapsed_ms) << ","
         << "\"limitations\":\"CPU sovereign LP and MILP Branch-and-Cut engine; GPU and QP are not "
            "implemented.\"";
    if (!error.empty()) {
        json << ",\"error\":\"" << json_escape(error) << "\"";
    }
    json << "}\n";
    const std::string payload = json.str();
    std::cout << payload;
    if (!output_path.empty()) {
        std::ofstream output(output_path);
        if (!output) {
            std::cerr << "cannot write output\n";
            return 8;
        }
        output << payload;
    }

    std::string timing_diag;
    if (resolved_engine == "pdlp") {
        if (backend_name == "gpu") {
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
                      ? (" [tol=" + json_number(pdlp_tolerance) + "]" + timing_diag)
                      : "")
              << (verified ? " VERIFIED\n" : " NOT VERIFIED\n");
    return exit_code(result.status);
}
