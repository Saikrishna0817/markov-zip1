#include "sihopt/foundation/build_info.hpp"
#include "sihopt/io/mps.hpp"
#include "sihopt/lp/reference/revised_simplex.hpp"
#include "sihopt/transform/canonicalize.hpp"
#include "sihopt/verify/primal_verifier.hpp"
#include "sihopt/verify/reference_lp_verifier.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int exit_code(sihopt::lp::reference::SolveStatus status) {
    using sihopt::lp::reference::SolveStatus;
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
    for (char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        if (c == '\n') {
            out += "\\n";
            continue;
        }
        out.push_back(c);
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
    out << "usage: sihopt-solve MODEL.mps [--output result.json] [--iteration-limit N]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    std::string output_path;
    sihopt::lp::reference::Options options;
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
        if (arg == "--iteration-limit") {
            if (i + 1 >= argc) {
                usage(std::cerr);
                return 8;
            }
            options.iteration_limit = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
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
    sihopt::verify::PrimalVerificationReport primal_report;
    sihopt::verify::ReferenceVerification canonical_report;
    sihopt::lp::reference::Result result;
    std::string error;

    try {
        const auto model = sihopt::io::parse_mps(input);
        const auto canonical = sihopt::transform::canonicalize(model);
        result = sihopt::lp::reference::solve(canonical, options);
        canonical_report = sihopt::verify::verify_reference_result(
            canonical, result, std::max(options.feasibility_tolerance, options.dual_tolerance));
        canonical_verified = canonical_report.accepted;
        if ((result.status == sihopt::lp::reference::SolveStatus::optimal ||
             result.status == sihopt::lp::reference::SolveStatus::infeasible ||
             result.status == sihopt::lp::reference::SolveStatus::unbounded) &&
            !canonical_verified) {
            result.status = sihopt::lp::reference::SolveStatus::numerical_failure;
            result.message = "canonical witness rejected: " + canonical_report.message;
        }
        if (result.status == sihopt::lp::reference::SolveStatus::optimal) {
            original_primal = sihopt::transform::reconstruct_primal(canonical, result.primal);
            original_objective = sihopt::transform::reconstruct_objective(canonical, result.objective);
            sihopt::verify::Candidate candidate{original_primal, original_objective};
            primal_report = sihopt::verify::verify_primal(model, candidate);
            original_verified = primal_report.passed;
            original_message = original_verified ? "original primal verified" : "original primal rejected";
            if (!original_verified) {
                result.status = sihopt::lp::reference::SolveStatus::numerical_failure;
                result.message = "original-model verification failed";
            }
        } else {
            original_message = "original primal not applicable";
        }
    } catch (const sihopt::io::MpsError& e) {
        result.status = sihopt::lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        error = e.what();
    } catch (const std::invalid_argument& e) {
        result.status = sihopt::lp::reference::SolveStatus::invalid_model;
        result.message = e.what();
        error = e.what();
    } catch (const std::length_error& e) {
        result.status = sihopt::lp::reference::SolveStatus::resource_limit;
        result.message = e.what();
        error = e.what();
    } catch (const std::exception& e) {
        result.status = sihopt::lp::reference::SolveStatus::numerical_failure;
        result.message = e.what();
        error = e.what();
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const bool verified = (result.status == sihopt::lp::reference::SolveStatus::optimal && original_verified &&
                           canonical_verified) ||
                          ((result.status == sihopt::lp::reference::SolveStatus::infeasible ||
                            result.status == sihopt::lp::reference::SolveStatus::unbounded) &&
                           canonical_verified);

    std::ostringstream json;
    json << "{\"version\":\"" << json_escape(std::string(sihopt::foundation::version())) << "\","
         << "\"milestone\":\"" << json_escape(std::string(sihopt::foundation::milestone())) << "\","
         << "\"status\":\"" << sihopt::lp::reference::to_string(result.status) << "\","
         << "\"verified\":" << (verified ? "true" : "false") << ","
         << "\"message\":\"" << json_escape(result.message) << "\","
         << "\"objective\":" << json_number(result.status == sihopt::lp::reference::SolveStatus::optimal
                                                ? original_objective
                                                : result.objective)
         << ","
         << "\"primal\":" << json_array(original_primal.empty() ? result.primal : original_primal) << ","
         << "\"canonical_verified\":" << (canonical_verified ? "true" : "false") << ","
         << "\"original_verified\":" << (original_verified ? "true" : "false") << ","
         << "\"original_message\":\"" << json_escape(original_message) << "\","
         << "\"maximum_primal_violation\":" << json_number(primal_report.maximum_row_violation) << ","
         << "\"maximum_variable_violation\":" << json_number(primal_report.maximum_variable_violation) << ","
         << "\"maximum_canonical_primal_violation\":" << json_number(canonical_report.maximum_primal_violation) << ","
         << "\"maximum_canonical_dual_violation\":" << json_number(canonical_report.maximum_dual_violation) << ","
         << "\"runtime_ms\":" << json_number(elapsed_ms) << ","
         << "\"phase_one_iterations\":" << result.phase_one_iterations << ","
         << "\"phase_two_iterations\":" << result.phase_two_iterations << ","
         << "\"limitations\":\"CPU continuous LP prototype; GPU, MILP, and QP are not implemented.\"";
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

    std::cerr << "SIHOpt " << sihopt::foundation::version() << " " << sihopt::lp::reference::to_string(result.status)
              << (verified ? " VERIFIED\n" : " NOT VERIFIED\n");
    return exit_code(result.status);
}
