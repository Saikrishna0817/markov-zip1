#include "markov_cero/lp/dual/dual_simplex.hpp"

#include "markov_cero/linalg/dense_lu.hpp"
#include "markov_cero/linalg/sparse_basis.hpp"
#include "markov_cero/verify/reference_lp_verifier.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace markov_cero::lp::dual {
namespace {

constexpr std::size_t maximum_rows = 1024;
constexpr std::size_t maximum_columns = 8192;
constexpr std::size_t maximum_iterations = 1000000;
constexpr std::size_t maximum_telemetry = 10000;
constexpr std::size_t maximum_dense_elements = 4U * 1024U * 1024U;
constexpr double maximum_tolerance = 1e-4;

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 255U;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex(std::uint64_t h) {
    std::ostringstream o;
    o << std::hex << std::setw(16) << std::setfill('0') << h;
    return o.str();
}

std::uint64_t hash_text(const std::string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

void check_product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::length_error("dual simplex size overflow");
    }
    if (a * b > maximum_dense_elements) {
        throw std::length_error("dual simplex dense workspace limit exceeded");
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += static_cast<long double>(a[i]) * b[i];
    }
    double v = static_cast<double>(s);
    if (!std::isfinite(v)) {
        throw std::overflow_error("non-finite dual simplex dot product");
    }
    return v;
}

linalg::SparseCsc sparse_basis_matrix(const transform::CanonicalModel& m,
                                      const std::vector<std::size_t>& basis) {
    std::vector<std::vector<double>> columns;
    columns.reserve(basis.size());
    for (auto j : basis) {
        std::vector<double> column(m.matrix.rows);
        for (std::size_t i = 0; i < m.matrix.rows; ++i) {
            column[i] = m.matrix(i, j);
        }
        columns.push_back(std::move(column));
    }
    return linalg::SparseCsc::from_columns(m.matrix.rows, columns);
}

linalg::SparseBasisOptions sparse_options(const Options& o) {
    linalg::SparseBasisOptions so;
    so.singular_tolerance = o.pivot_tolerance;
    so.update_pivot_tolerance = o.pivot_tolerance;
    so.maximum_dimension = maximum_rows;
    so.maximum_nonzeros = maximum_dense_elements;
    so.maximum_factor_nonzeros = maximum_dense_elements;
    so.maximum_updates = 64;
    so.eta_density_trigger = 0.5;
    return so;
}

bool significant_negative_reduced_cost(const transform::CanonicalModel& m, std::size_t j,
                                       const std::vector<double>& y, double rc, double tol) {
    long double scale = std::abs(static_cast<long double>(m.objective[j]));
    for (std::size_t i = 0; i < m.matrix.rows; ++i) {
        scale += std::abs(static_cast<long double>(m.matrix(i, j)) * y[i]);
    }
    const double z = static_cast<double>(scale);
    const double allowed =
        tol * z + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, z);
    return rc < -allowed;
}

std::vector<double> full_primal(std::size_t n, const std::vector<std::size_t>& basis,
                                const std::vector<double>& xb) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        x[basis[i]] = xb[i];
    }
    return x;
}

void validate_options(const Options& o) {
    if (!o.iteration_limit || o.iteration_limit > maximum_iterations ||
        o.telemetry_limit > maximum_telemetry || !std::isfinite(o.feasibility_tolerance) ||
        !std::isfinite(o.dual_tolerance) || !std::isfinite(o.pivot_tolerance) ||
        !std::isfinite(o.condition_trigger) || o.feasibility_tolerance <= 0 ||
        o.dual_tolerance <= 0 || o.pivot_tolerance <= 0 || o.condition_trigger < 0 ||
        o.feasibility_tolerance > maximum_tolerance || o.dual_tolerance > maximum_tolerance ||
        o.pivot_tolerance > maximum_tolerance || o.pivot_tolerance > o.feasibility_tolerance ||
        o.condition_trigger > 1) {
        throw std::invalid_argument("invalid dual simplex options");
    }
}

void validate_basis(const transform::CanonicalModel& m, const BasisState& s) {
    if (s.rows != m.matrix.rows || s.columns != m.matrix.columns ||
        s.model_fingerprint != fingerprint(m) || s.basic_variables.size() != m.matrix.rows) {
        throw std::invalid_argument("warm basis metadata mismatch");
    }
    std::vector<bool> seen(m.matrix.columns);
    for (auto j : s.basic_variables) {
        if (j >= m.matrix.columns || seen[j]) {
            throw std::invalid_argument("warm basis index invalid or duplicate");
        }
        seen[j] = true;
    }
    if (m.matrix.rows > m.matrix.columns) {
        throw std::invalid_argument("warm basis cannot be square");
    }
    try {
        (void)linalg::SparseLu::factorize(sparse_basis_matrix(m, s.basic_variables));
    } catch (const std::exception&) {
        throw std::invalid_argument("warm basis is singular");
    }
}

Result cold(const transform::CanonicalModel& m, const Options& o, const std::string& why) {
    Result out;
    reference::Options ro;
    ro.iteration_limit = o.iteration_limit;
    ro.telemetry_limit = o.telemetry_limit;
    ro.feasibility_tolerance = o.feasibility_tolerance;
    ro.dual_tolerance = o.dual_tolerance;
    ro.pivot_tolerance = o.pivot_tolerance;
    auto r = reference::solve(m, ro);
    out.solution = std::move(r);
    out.used_cold_fallback = true;
    out.message = why;
    if (out.solution.status == reference::SolveStatus::optimal &&
        out.solution.basis.size() == m.matrix.rows) {
        out.basis_state = make_basis_state(m, out.solution.basis);
    }
    return out;
}

std::size_t select_leaving_row(const transform::CanonicalModel& m,
                               linalg::SparseBasisFactorization& factor,
                               const std::vector<double>& xb, const std::vector<std::size_t>& basis,
                               const Options& o, double& worst) {
    std::size_t leaving = m.matrix.rows;
    worst = 0;
    for (std::size_t i = 0; i < m.matrix.rows; ++i) {
        if (xb[i] >= -o.feasibility_tolerance) {
            continue;
        }
        if (o.pricing == PricingPolicy::bland) {
            if (leaving == m.matrix.rows || basis[i] < basis[leaving]) {
                leaving = i;
            }
            continue;
        }
        // Tableau-norm weight ||A^T B^{-T} e_i||^2; not conventional DSE ||B^{-T} e_i||^2.
        // Full weight is recomputed for every candidate row (O(m) solves + SpMV per pivot).
        // Incremental Forrest–Goldfarb updates are deferred until MILP node reoptimization (M9).
        std::vector<double> e(m.matrix.rows);
        e[i] = 1;
        auto pi = factor.solve_transpose(e);
        auto row = linalg::multiply_transpose(m.matrix, pi);
        const double weight = dot(row, row);
        const double score =
            (-xb[i]) / std::sqrt(std::max(weight, std::numeric_limits<double>::min()));
        if (leaving == m.matrix.rows || score > worst ||
            (score == worst && basis[i] < basis[leaving])) {
            leaving = i;
            worst = score;
        }
    }
    return leaving;
}

std::size_t select_entering_column(const transform::CanonicalModel& m,
                                   const std::vector<double>& alpha, const std::vector<double>& rc,
                                   const std::vector<bool>& is_basic, const Options& o,
                                   double& best_ratio) {
    std::size_t entering = m.matrix.columns;
    best_ratio = std::numeric_limits<double>::infinity();
    double limit = std::numeric_limits<double>::infinity();
    if (o.harris_ratio) {
        for (std::size_t j = 0; j < m.matrix.columns; ++j) {
            if (is_basic[j] || alpha[j] >= -o.pivot_tolerance) {
                continue;
            }
            const double q = (std::max(0.0, rc[j]) + o.dual_tolerance) / (-alpha[j]);
            if (!std::isfinite(q)) {
                throw std::overflow_error("non-finite Harris ratio");
            }
            limit = std::min(limit, q);
        }
    }
    double best_pivot = 0;
    for (std::size_t j = 0; j < m.matrix.columns; ++j) {
        if (is_basic[j] || alpha[j] >= -o.pivot_tolerance) {
            continue;
        }
        const double q = std::max(0.0, rc[j]) / (-alpha[j]);
        if (!std::isfinite(q)) {
            throw std::overflow_error("non-finite dual ratio");
        }
        if (o.harris_ratio) {
            if (q <= limit && (entering == m.matrix.columns || -alpha[j] > best_pivot ||
                               (-alpha[j] == best_pivot && j < entering))) {
                entering = j;
                best_ratio = q;
                best_pivot = -alpha[j];
            }
        } else if (q < best_ratio || (q == best_ratio && j < entering)) {
            entering = j;
            best_ratio = q;
            best_pivot = -alpha[j];
        }
    }
    return entering;
}

Result certified_optimal(const transform::CanonicalModel& m, const std::vector<std::size_t>& basis,
                         const std::vector<double>& xb, const std::vector<double>& y,
                         const Options& o) {
    Result out;
    out.solution.status = reference::SolveStatus::optimal;
    out.solution.primal = full_primal(m.matrix.columns, basis, xb);
    out.solution.dual = y;
    out.solution.basis = basis;
    out.solution.objective = dot(m.objective, out.solution.primal) + m.objective_offset;
    out.solution.message = "dual revised simplex optimum";
    out.basis_state = make_basis_state(m, basis);
    auto check = verify::verify_reference_result(
        m, out.solution, std::max(o.feasibility_tolerance, o.dual_tolerance));
    if (!check.accepted) {
        out.solution.status = reference::SolveStatus::numerical_failure;
        out.solution.message = "dual optimum witness rejected: " + check.message;
    }
    out.message = out.solution.message;
    return out;
}

Result certified_farkas(const transform::CanonicalModel& m, const std::vector<double>& pi,
                        const Options& o) {
    Result out;
    out.solution.status = reference::SolveStatus::infeasible;
    out.solution.certificate.resize(m.matrix.rows);
    for (std::size_t i = 0; i < m.matrix.rows; ++i) {
        out.solution.certificate[i] = -pi[i];
    }
    out.solution.message = "dual simplex Farkas certificate";
    auto check = verify::verify_reference_result(
        m, out.solution, std::max(o.feasibility_tolerance, o.dual_tolerance));
    if (!check.accepted) {
        out.solution.status = reference::SolveStatus::numerical_failure;
        out.solution.message = "dual Farkas witness rejected: " + check.message;
    }
    out.message = out.solution.message;
    return out;
}

} // namespace

std::string fingerprint(const transform::CanonicalModel& m) {
    m.validate();
    std::uint64_t h = 1469598103934665603ULL;
    h = mix(h, m.matrix.rows);
    h = mix(h, m.matrix.columns);
    for (double v : m.matrix.values) {
        h = mix(h, std::bit_cast<std::uint64_t>(v));
    }
    for (double v : m.objective) {
        h = mix(h, std::bit_cast<std::uint64_t>(v));
    }
    h = mix(h, std::bit_cast<std::uint64_t>(m.objective_offset));
    return hex(h);
}

BasisState make_basis_state(const transform::CanonicalModel& m, const std::vector<std::size_t>& b) {
    BasisState s{m.matrix.rows, m.matrix.columns, fingerprint(m), b};
    validate_basis(m, s);
    (void)linalg::SparseLu::factorize(sparse_basis_matrix(m, b));
    return s;
}

void validate_basis_artifact(const BasisState& s) {
    if (s.rows > maximum_rows || s.columns > maximum_columns || s.rows > s.columns ||
        s.basic_variables.size() != s.rows || s.model_fingerprint.size() != 16) {
        throw std::invalid_argument("invalid basis artifact metadata");
    }
    for (char c : s.model_fingerprint) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) {
            throw std::invalid_argument("invalid basis fingerprint");
        }
    }
    std::vector<bool> seen(s.columns);
    for (auto j : s.basic_variables) {
        if (j >= s.columns || seen[j]) {
            throw std::invalid_argument("basis artifact index invalid or duplicate");
        }
        seen[j] = true;
    }
}

std::string serialize_basis(const BasisState& s) {
    validate_basis_artifact(s);
    std::ostringstream body;
    body << "MARKOV-CERO-BASIS-1 " << s.rows << ' ' << s.columns << ' ' << s.model_fingerprint
         << ' ' << s.basic_variables.size();
    for (auto j : s.basic_variables) {
        body << ' ' << j;
    }
    const auto text = body.str();
    return text + ' ' + hex(hash_text(text)) + "\n";
}

BasisState parse_basis(const std::string& text) {
    std::istringstream in(text);
    std::string magic, fp, checksum, trailing;
    BasisState s;
    std::size_t count = 0;
    if (!(in >> magic >> s.rows >> s.columns >> fp >> count) || magic != "MARKOV-CERO-BASIS-1" ||
        count > maximum_rows) {
        throw std::invalid_argument("invalid basis header");
    }
    s.model_fingerprint = fp;
    s.basic_variables.resize(count);
    for (auto& j : s.basic_variables) {
        if (!(in >> j)) {
            throw std::invalid_argument("truncated basis");
        }
    }
    if (!(in >> checksum) || (in >> trailing)) {
        throw std::invalid_argument("invalid basis trailer");
    }
    std::ostringstream body;
    body << magic << ' ' << s.rows << ' ' << s.columns << ' ' << fp << ' ' << count;
    for (auto j : s.basic_variables) {
        body << ' ' << j;
    }
    if (checksum != hex(hash_text(body.str()))) {
        throw std::invalid_argument("basis checksum mismatch");
    }
    validate_basis_artifact(s);
    return s;
}

Result solve(const transform::CanonicalModel& m, const Options& o,
             const std::optional<BasisState>& warm) {
    Result out;
    bool validating_warm = false;
    try {
        m.validate();
    } catch (const std::exception& e) {
        out.solution.status = reference::SolveStatus::invalid_model;
        out.solution.message = e.what();
        out.message = e.what();
        return out;
    }
    try {
        validate_options(o);
    } catch (const std::exception& e) {
        out.solution.status = reference::SolveStatus::invalid_options;
        out.solution.message = e.what();
        out.message = e.what();
        return out;
    }
    try {
        if (m.matrix.rows > maximum_rows || m.matrix.columns > maximum_columns) {
            throw std::length_error("dual simplex reference dimension limit exceeded");
        }
        check_product(m.matrix.rows, m.matrix.rows);
        if (!warm) {
            return cold(m, o, "cold solve delegated to certified M3 oracle");
        }
        validating_warm = true;
        validate_basis(m, *warm);
        validating_warm = false;
        out.used_warm_start = true;
        out.telemetry.reserve(std::min(o.iteration_limit, o.telemetry_limit));
        auto basis = warm->basic_variables;
        auto factor = linalg::SparseBasisFactorization::factorize(sparse_basis_matrix(m, basis),
                                                                  sparse_options(o));
        out.refactorizations = factor.statistics().refactorizations;
        for (std::size_t step = 0; step < o.iteration_limit; ++step) {
            const auto& diagnostics = factor.diagnostics();
            if (m.matrix.rows > 0 && diagnostics.maximum_absolute_pivot > 0 &&
                diagnostics.minimum_absolute_pivot / diagnostics.maximum_absolute_pivot <
                    o.condition_trigger) {
                throw std::runtime_error("basis condition trigger reached");
            }
            auto xb = factor.solve(m.rhs);
            std::vector<double> cb(m.matrix.rows);
            std::vector<bool> is_basic(m.matrix.columns);
            for (std::size_t i = 0; i < m.matrix.rows; ++i) {
                cb[i] = m.objective[basis[i]];
                is_basic[basis[i]] = true;
            }
            auto y = factor.solve_transpose(cb);
            auto aty = linalg::multiply_transpose(m.matrix, y);
            std::vector<double> rc(m.matrix.columns);
            for (std::size_t j = 0; j < m.matrix.columns; ++j) {
                rc[j] = m.objective[j] - aty[j];
                if (significant_negative_reduced_cost(m, j, y, rc[j], o.dual_tolerance)) {
                    if (step == 0) {
                        if (o.allow_cold_fallback) {
                            return cold(m, o, "warm basis is not dual feasible; cold fallback");
                        }
                        throw std::runtime_error("warm basis is not dual feasible");
                    }
                    throw std::runtime_error("dual feasibility lost after pivot");
                }
            }
            double worst = 0;
            const std::size_t leaving = select_leaving_row(m, factor, xb, basis, o, worst);
            if (leaving == m.matrix.rows) {
                auto certified = certified_optimal(m, basis, xb, y, o);
                certified.used_warm_start = true;
                certified.telemetry = std::move(out.telemetry);
                certified.telemetry_truncated = out.telemetry_truncated;
                certified.refactorizations = out.refactorizations;
                return certified;
            }
            std::vector<double> e(m.matrix.rows);
            e[leaving] = 1;
            auto pi = factor.solve_transpose(e);
            auto alpha = linalg::multiply_transpose(m.matrix, pi);
            double best_ratio = 0;
            const std::size_t entering =
                select_entering_column(m, alpha, rc, is_basic, o, best_ratio);
            if (entering == m.matrix.columns) {
                auto certified = certified_farkas(m, pi, o);
                certified.used_warm_start = true;
                certified.telemetry = std::move(out.telemetry);
                certified.telemetry_truncated = out.telemetry_truncated;
                certified.refactorizations = out.refactorizations;
                return certified;
            }
            if (out.telemetry.size() < o.telemetry_limit) {
                out.telemetry.push_back({step, dot(cb, xb) + m.objective_offset, xb[leaving],
                                         basis[leaving], entering, alpha[entering],
                                         o.harris_ratio});
            } else {
                out.telemetry_truncated = true;
            }
            std::vector<double> entering_column(m.matrix.rows);
            for (std::size_t i = 0; i < m.matrix.rows; ++i) {
                entering_column[i] = m.matrix(i, entering);
            }
            factor.replace_column(leaving, entering_column);
            basis[leaving] = entering;
            if (factor.needs_refactorization()) {
                factor.refactorize();
            }
            out.refactorizations = factor.statistics().refactorizations;
        }
        out.solution.status = reference::SolveStatus::iteration_limit;
        out.solution.message = "dual simplex iteration limit";
        out.message = out.solution.message;
        return out;
    } catch (const std::length_error& e) {
        out.solution.status = reference::SolveStatus::resource_limit;
        out.solution.message = e.what();
        out.message = e.what();
        return out;
    } catch (const std::invalid_argument& e) {
        if (validating_warm && warm && o.allow_cold_fallback) {
            return cold(m, o, std::string("invalid warm start; cold fallback: ") + e.what());
        }
        out.solution.status = reference::SolveStatus::numerical_failure;
        out.solution.message = e.what();
        out.message = e.what();
        return out;
    } catch (const std::exception& e) {
        out.solution.status = reference::SolveStatus::numerical_failure;
        out.solution.message = e.what();
        out.message = e.what();
        return out;
    }
}

} // namespace markov_cero::lp::dual
