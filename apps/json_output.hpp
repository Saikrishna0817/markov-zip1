#pragma once

#include "markov_cero/foundation/build_info.hpp"
#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/verify/primal_verifier.hpp"
#include "markov_cero/verify/reference_lp_verifier.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace markov_cero::apps {

inline int exit_code(markov_cero::lp::reference::SolveStatus status) {
    using markov_cero::lp::reference::SolveStatus;
    switch (status) {
    case SolveStatus::optimal: return 0;
    case SolveStatus::infeasible: return 1;
    case SolveStatus::unbounded: return 2;
    case SolveStatus::invalid_model: return 3;
    case SolveStatus::invalid_options: return 4;
    case SolveStatus::resource_limit: return 5;
    case SolveStatus::iteration_limit: return 6;
    case SolveStatus::numerical_failure: return 7;
    }
    return 7;
}

inline std::string format_violation(
    const markov_cero::verify::PrimalVerificationReport& report) {
    if (report.violations.empty()) return "";
    const auto& v = report.violations[0];
    return v.category + " idx=" + std::to_string(v.index) +
           " act=" + std::to_string(v.actual) +
           " bnd=" + std::to_string(v.bound) +
           " diff=" + std::to_string(v.magnitude) +
           " allow=" + std::to_string(v.allowance);
}

inline std::string json_escape(std::string_view text) {
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

inline std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream o;
    o.setf(std::ios::fmtflags(0), std::ios::floatfield);
    o.precision(17);
    o << value;
    return o.str();
}

inline std::string json_array(const std::vector<double>& values) {
    std::ostringstream o;
    o << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) o << ',';
        o << json_number(values[i]);
    }
    o << ']';
    return o.str();
}

struct JsonOutputData {
    std::string resolved_engine;
    markov_cero::lp::reference::Result result;
    std::size_t model_rows = 0;
    std::size_t model_cols = 0;
    std::size_t model_nnz = 0;
    bool verified = false;
    double original_objective = 0.0;
    std::vector<double> original_primal;
    bool canonical_verified = false;
    bool original_verified = false;
    std::string original_message;
    bool used_warm_start = false;
    bool used_cold_fallback = false;
    markov_cero::verify::PrimalVerificationReport primal_report;
    markov_cero::verify::ReferenceVerification canonical_report;
    double elapsed_ms = 0.0;
    std::size_t nodes_explored = 0;
    std::size_t total_lp_iterations = 0;
    double best_bound = 0.0;
    double relative_gap = 0.0;
    std::size_t cuts_generated = 0;
    std::size_t heuristics_found = 0;
    double pdlp_tolerance = 0.0;
    double pdlp_res_primal_infeas = 0.0;
    double pdlp_res_dual_infeas = 0.0;
    double pdlp_res_gap = 0.0;
    std::string backend_name;
    double pdlp_h2d_ms = 0.0;
    double pdlp_kernel_ms = 0.0;
    double pdlp_d2h_ms = 0.0;
    double pdlp_total_ms = 0.0;
    std::string error;
    std::string output_path;
};

inline void emit_json_output(const JsonOutputData& data) {
    std::ostringstream json;
    json << "{\"version\":\"" << json_escape(std::string(markov_cero::foundation::version()))
         << "\","
         << "\"milestone\":\"" << json_escape(std::string(markov_cero::foundation::milestone()))
         << "\","
         << "\"engine\":\"" << json_escape(data.resolved_engine) << "\","
         << "\"status\":\"" << markov_cero::lp::reference::to_string(data.result.status) << "\","
         << "\"rows\":" << data.model_rows << ","
         << "\"cols\":" << data.model_cols << ","
         << "\"nonzeros\":" << data.model_nnz << ","
         << "\"verified\":" << (data.verified ? "true" : "false") << ","
         << "\"message\":\"" << json_escape(data.result.message) << "\","
         << "\"objective\":"
         << json_number(data.result.status == markov_cero::lp::reference::SolveStatus::optimal
                            ? data.original_objective
                            : data.result.objective)
         << ","
         << "\"primal\":"
         << json_array(data.original_primal.empty() ? data.result.primal
                                                    : data.original_primal)
         << ","
         << "\"canonical_verified\":" << (data.canonical_verified ? "true" : "false") << ","
         << "\"original_verified\":" << (data.original_verified ? "true" : "false") << ","
         << "\"original_message\":\"" << json_escape(data.original_message) << "\","
         << "\"used_warm_start\":" << (data.used_warm_start ? "true" : "false") << ","
         << "\"used_cold_fallback\":" << (data.used_cold_fallback ? "true" : "false") << ","
         << "\"maximum_primal_violation\":" << json_number(data.primal_report.maximum_row_violation)
         << ","
         << "\"maximum_variable_violation\":"
         << json_number(data.primal_report.maximum_variable_violation) << ","
         << "\"maximum_integrality_violation\":"
         << json_number(data.primal_report.maximum_integrality_violation) << ","
         << "\"maximum_canonical_primal_violation\":"
         << json_number(data.canonical_report.maximum_primal_violation) << ","
         << "\"maximum_canonical_dual_violation\":"
         << json_number(data.canonical_report.maximum_dual_violation) << ","
         << "\"runtime_ms\":" << json_number(data.elapsed_ms) << ","
         << "\"nodes_explored\":" << data.nodes_explored << ","
         << "\"lp_iterations\":" << data.total_lp_iterations << ","
         << "\"best_bound\":" << json_number(data.best_bound) << ","
         << "\"relative_gap\":" << json_number(data.relative_gap) << ","
         << "\"cuts_generated\":" << data.cuts_generated << ","
         << "\"heuristics_found\":" << data.heuristics_found << ","
         << "\"phase_one_iterations\":" << data.result.phase_one_iterations << ","
         << "\"phase_two_iterations\":" << data.result.phase_two_iterations << ","
         << "\"pdlp_tolerance\":" << json_number(data.pdlp_tolerance) << ","
         << "\"relative_primal_residual\":" << json_number(data.pdlp_res_primal_infeas) << ","
         << "\"relative_dual_residual\":" << json_number(data.pdlp_res_dual_infeas) << ","
         << "\"relative_duality_gap\":" << json_number(data.pdlp_res_gap) << ","
         << "\"backend\":\"" << json_escape(data.backend_name) << "\","
         << "\"h2d_ms\":" << json_number(data.pdlp_h2d_ms) << ","
         << "\"kernel_ms\":" << json_number(data.pdlp_kernel_ms) << ","
         << "\"d2h_ms\":" << json_number(data.pdlp_d2h_ms) << ","
         << "\"total_ms\":"
         << json_number(data.resolved_engine == "pdlp" ? data.pdlp_total_ms
                                                       : data.elapsed_ms)
         << ","
         << "\"limitations\":\"Sovereign LP/MILP/QP/MIQP (CPU/GPU) engine.\"";
    if (!data.error.empty()) {
        json << ",\"error\":\"" << json_escape(data.error) << "\"";
    }
    json << "}\n";
    const std::string payload = json.str();
    std::cout << payload;
    if (!data.output_path.empty()) {
        std::ofstream output(data.output_path);
        if (output) {
            output << payload;
        } else {
            std::cerr << "cannot write output\n";
        }
    }
}

} // namespace markov_cero::apps
