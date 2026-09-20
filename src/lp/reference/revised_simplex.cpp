#include "markov_cero/lp/reference/revised_simplex.hpp"

#include "markov_cero/linalg/dense_lu.hpp"
#include "markov_cero/verify/reference_lp_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace markov_cero::lp::reference {
namespace {

constexpr std::size_t maximum_rows = 1024;
constexpr std::size_t maximum_columns = 8192;
constexpr std::size_t maximum_expanded_elements = 4U * 1024U * 1024U;
constexpr std::size_t maximum_iterations = 1000000;
constexpr std::size_t maximum_telemetry = 10000;
constexpr double maximum_tolerance = 1e-4;

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::length_error("simplex dimension addition overflow");
    }
    return a + b;
}

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::length_error("simplex dimension product overflow");
    }
    const auto n = a * b;
    if (n > maximum_expanded_elements) {
        throw std::length_error("simplex dense workspace limit exceeded");
    }
    return n;
}

struct Work {
    std::size_t rows{};
    std::size_t original_rows{};
    std::size_t original_columns{};
    std::size_t total_columns{};
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> row_sign;
    std::vector<std::size_t> row_origin;
    std::vector<std::size_t> basis;
};

std::vector<double> column(const Work& w, std::size_t j) {
    std::vector<double> v(w.rows);
    for (std::size_t i = 0; i < w.rows; ++i) {
        v[i] = w.a[i * w.total_columns + j];
    }
    return v;
}

linalg::DenseMatrix basis_matrix(const Work& w) {
    linalg::DenseMatrix b{w.rows, w.rows, std::vector<double>(w.rows * w.rows)};
    for (std::size_t j = 0; j < w.rows; ++j) {
        for (std::size_t i = 0; i < w.rows; ++i) {
            b.values[i * w.rows + j] = w.a[i * w.total_columns + w.basis[j]];
        }
    }
    return b;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += static_cast<long double>(a[i]) * b[i];
    }
    const double d = static_cast<double>(s);
    if (!std::isfinite(d)) {
        throw std::overflow_error("non-finite simplex dot product");
    }
    return d;
}

struct IterationOutcome {
    SolveStatus status{SolveStatus::numerical_failure};
    std::vector<double> xb;
    std::vector<double> y;
    std::vector<double> ray;
    std::size_t iterations{};
};

bool significant_negative_reduced_cost(const Work& w, std::size_t j,
                                       const std::vector<double>& cost,
                                       const std::vector<double>& y, double rc, double tol) {
    long double scale = std::abs(static_cast<long double>(cost[j]));
    for (std::size_t i = 0; i < w.rows; ++i) {
        scale += std::abs(static_cast<long double>(w.a[i * w.total_columns + j]) * y[i]);
    }
    const double z = static_cast<double>(scale);
    const double allowed =
        tol * z + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, z);
    return rc < -allowed;
}

void snap_basic_solution(std::vector<double>& xb, double feasibility_tolerance) {
    for (double& v : xb) {
        if (v < 0 && v >= -feasibility_tolerance) {
            v = 0;
        }
        if (v < 0) {
            throw std::runtime_error("primal basis lost feasibility");
        }
    }
}

std::size_t select_entering(const Work& w, const std::vector<double>& cost,
                            const std::vector<double>& y, const std::vector<bool>& basic,
                            std::size_t enter_limit, const Options& o, double& minimum_rc) {
    std::size_t entering = enter_limit;
    minimum_rc = 0;
    for (std::size_t j = 0; j < enter_limit; ++j) {
        if (basic[j]) {
            continue;
        }
        const double rc = cost[j] - dot(column(w, j), y);
        if (!significant_negative_reduced_cost(w, j, cost, y, rc, o.dual_tolerance)) {
            continue;
        }
        if (o.bland_anti_cycling) {
            entering = j;
            minimum_rc = rc;
            break;
        }
        if (entering == enter_limit || rc < minimum_rc) {
            entering = j;
            minimum_rc = rc;
        }
    }
    return entering;
}

std::size_t select_leaving(const Work& w, const std::vector<double>& xb,
                           const std::vector<double>& d, const Options& o, double& theta) {
    std::size_t leaving_row = w.rows;
    theta = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < w.rows; ++i) {
        if (d[i] <= o.pivot_tolerance) {
            continue;
        }
        const double ratio = xb[i] / d[i];
        if (!std::isfinite(ratio)) {
            throw std::overflow_error("non-finite simplex ratio");
        }
        if (ratio < theta ||
            (ratio == theta && (leaving_row == w.rows || w.basis[i] < w.basis[leaving_row]))) {
            theta = ratio;
            leaving_row = i;
        }
    }
    return leaving_row;
}

IterationOutcome iterate(Work& w, const std::vector<double>& cost, std::size_t enter_limit,
                         const Options& o, int phase, std::size_t budget,
                         std::vector<IterationRecord>& log, bool& telemetry_truncated) {
    IterationOutcome out;
    for (std::size_t step = 0; step <= budget; ++step) {
        if (step == budget) {
            out.status = SolveStatus::iteration_limit;
            out.iterations = step;
            return out;
        }
        auto lu = linalg::DenseLu::factorize(basis_matrix(w), o.pivot_tolerance);
        auto xb = lu.solve(w.b);
        snap_basic_solution(xb, o.feasibility_tolerance);
        std::vector<double> cb(w.rows);
        std::vector<bool> basic(w.total_columns);
        for (std::size_t i = 0; i < w.rows; ++i) {
            cb[i] = cost[w.basis[i]];
            basic[w.basis[i]] = true;
        }
        auto y = lu.solve_transpose(cb);
        double minimum_rc = 0;
        const std::size_t entering = select_entering(w, cost, y, basic, enter_limit, o, minimum_rc);
        if (entering == enter_limit) {
            out.status = SolveStatus::optimal;
            out.xb = std::move(xb);
            out.y = std::move(y);
            out.iterations = step;
            return out;
        }
        auto d = lu.solve(column(w, entering));
        double theta = 0;
        const std::size_t leaving_row = select_leaving(w, xb, d, o, theta);
        if (leaving_row == w.rows) {
            for (double value : d) {
                if (value > 0) {
                    out.status = SolveStatus::numerical_failure;
                    out.iterations = step;
                    return out;
                }
            }
            out.status = SolveStatus::unbounded;
            out.xb = std::move(xb);
            out.y = std::move(y);
            out.ray.assign(w.total_columns, 0);
            out.ray[entering] = 1;
            for (std::size_t i = 0; i < w.rows; ++i) {
                out.ray[w.basis[i]] = -d[i];
            }
            out.iterations = step;
            return out;
        }
        const auto leaving = w.basis[leaving_row];
        if (log.size() < o.telemetry_limit) {
            log.push_back({log.size(), phase, dot(cb, xb), minimum_rc, entering, leaving,
                           theta <= o.feasibility_tolerance});
        } else {
            telemetry_truncated = true;
        }
        w.basis[leaving_row] = entering;
    }
    return out;
}

Work make_work(const transform::CanonicalModel& m) {
    Work w;
    w.rows = m.matrix.rows;
    w.original_rows = w.rows;
    w.original_columns = m.matrix.columns;
    if (w.rows > maximum_rows || w.original_columns > maximum_columns) {
        throw std::length_error("simplex reference dimension limit exceeded");
    }
    w.total_columns = checked_add(w.original_columns, w.original_rows);
    const auto work_elements = checked_product(w.rows, w.total_columns);
    checked_product(w.rows, w.rows);
    w.a.assign(work_elements, 0);
    w.b = m.rhs;
    w.row_sign.assign(w.rows, 1);
    w.row_origin.resize(w.rows);
    for (std::size_t i = 0; i < w.rows; ++i) {
        w.row_origin[i] = i;
        if (w.b[i] < 0) {
            w.b[i] = -w.b[i];
            w.row_sign[i] = -1;
        }
        for (std::size_t j = 0; j < w.original_columns; ++j) {
            w.a[i * w.total_columns + j] = w.row_sign[i] * m.matrix(i, j);
        }
        w.a[i * w.total_columns + w.original_columns + i] = 1;
    }
    return w;
}

bool crash_basis(Work& w, double tol) {
    w.basis.assign(w.rows, w.total_columns);
    std::vector<bool> used(w.original_columns);
    for (std::size_t i = 0; i < w.rows; ++i) {
        for (std::size_t j = 0; j < w.original_columns; ++j) {
            if (used[j]) {
                continue;
            }
            bool unit = true;
            for (std::size_t r = 0; r < w.rows; ++r) {
                const double expected = r == i ? 1.0 : 0.0;
                if (std::abs(w.a[r * w.total_columns + j] - expected) > tol) {
                    unit = false;
                    break;
                }
            }
            if (unit) {
                w.basis[i] = j;
                used[j] = true;
                break;
            }
        }
        if (w.basis[i] == w.total_columns) {
            return false;
        }
    }
    return true;
}

void remove_row(Work& w, std::size_t victim) {
    std::vector<double> next;
    next.reserve((w.rows - 1) * w.total_columns);
    for (std::size_t i = 0; i < w.rows; ++i) {
        if (i == victim) {
            continue;
        }
        next.insert(next.end(), w.a.begin() + static_cast<std::ptrdiff_t>(i * w.total_columns),
                    w.a.begin() + static_cast<std::ptrdiff_t>((i + 1) * w.total_columns));
    }
    w.a = std::move(next);
    w.b.erase(w.b.begin() + static_cast<std::ptrdiff_t>(victim));
    w.row_sign.erase(w.row_sign.begin() + static_cast<std::ptrdiff_t>(victim));
    w.row_origin.erase(w.row_origin.begin() + static_cast<std::ptrdiff_t>(victim));
    w.basis.erase(w.basis.begin() + static_cast<std::ptrdiff_t>(victim));
    --w.rows;
}

void remove_artificials(Work& w, double tol) {
    for (std::size_t i = 0; i < w.rows;) {
        if (w.basis[i] < w.original_columns) {
            ++i;
            continue;
        }
        auto lu = linalg::DenseLu::factorize(basis_matrix(w), tol);
        std::vector<bool> basic(w.total_columns);
        for (auto j : w.basis) {
            basic[j] = true;
        }
        std::size_t entering = w.original_columns;
        for (std::size_t j = 0; j < w.original_columns; ++j) {
            if (basic[j]) {
                continue;
            }
            auto d = lu.solve(column(w, j));
            if (std::abs(d[i]) > tol) {
                entering = j;
                break;
            }
        }
        if (entering < w.original_columns) {
            w.basis[i] = entering;
            ++i;
        } else {
            remove_row(w, i);
        }
    }
}

Result certify(const transform::CanonicalModel& m, Result r, double tolerance) {
    if (r.status == SolveStatus::optimal || r.status == SolveStatus::infeasible ||
        r.status == SolveStatus::unbounded) {
        const auto report = verify::verify_reference_result(m, r, tolerance);
        if (!report.accepted) {
            r.status = SolveStatus::numerical_failure;
            r.message = "internal witness verification failed: " + report.message;
        }
    }
    return r;
}

std::vector<double> full_solution(const Work& w, const std::vector<double>& xb) {
    std::vector<double> x(w.total_columns);
    for (std::size_t i = 0; i < w.rows; ++i) {
        x[w.basis[i]] = xb[i];
    }
    return x;
}

bool options_invalid(const Options& o) {
    return o.iteration_limit == 0 || !std::isfinite(o.feasibility_tolerance) ||
           !std::isfinite(o.dual_tolerance) || !std::isfinite(o.pivot_tolerance) ||
           o.feasibility_tolerance <= 0 || o.dual_tolerance <= 0 || o.pivot_tolerance <= 0 ||
           o.feasibility_tolerance > maximum_tolerance || o.dual_tolerance > maximum_tolerance ||
           o.pivot_tolerance > maximum_tolerance || o.pivot_tolerance > o.feasibility_tolerance ||
           o.iteration_limit > maximum_iterations || o.telemetry_limit > maximum_telemetry;
}

} // namespace

Result solve(const transform::CanonicalModel& m, const Options& o) {
    Result result;
    try {
        m.validate();
    } catch (const std::exception& e) {
        result.status = SolveStatus::invalid_model;
        result.message = e.what();
        return result;
    }
    if (options_invalid(o)) {
        result.status = SolveStatus::invalid_options;
        result.message = "invalid simplex options";
        return result;
    }
    try {
        auto w = make_work(m);
        result.telemetry.reserve(std::min(o.iteration_limit, o.telemetry_limit));
        std::vector<double> phase_two_cost(w.total_columns);
        std::copy(m.objective.begin(), m.objective.end(), phase_two_cost.begin());
        std::size_t remaining = o.iteration_limit;
        if (!crash_basis(w, o.pivot_tolerance)) {
            w.basis.resize(w.rows);
            std::vector<double> phase_one_cost(w.total_columns);
            for (std::size_t i = 0; i < w.rows; ++i) {
                w.basis[i] = w.original_columns + i;
                phase_one_cost[w.basis[i]] = 1;
            }
            auto one = iterate(w, phase_one_cost, w.total_columns, o, 1, remaining,
                               result.telemetry, result.telemetry_truncated);
            result.phase_one_iterations = one.iterations;
            remaining -= std::min(remaining, one.iterations);
            if (one.status == SolveStatus::iteration_limit) {
                result.status = one.status;
                result.message = "phase I iteration limit";
                return result;
            }
            if (one.status != SolveStatus::optimal) {
                result.status = SolveStatus::numerical_failure;
                result.message = "phase I failed";
                return result;
            }
            auto one_x = full_solution(w, one.xb);
            const double phase_one_objective = dot(phase_one_cost, one_x);
            if (phase_one_objective > o.feasibility_tolerance) {
                result.status = SolveStatus::infeasible;
                result.certificate.assign(w.original_rows, 0);
                for (std::size_t i = 0; i < w.rows; ++i) {
                    result.certificate[w.row_origin[i]] = w.row_sign[i] * one.y[i];
                }
                result.message = "validated phase I Farkas candidate";
                return certify(m, std::move(result),
                               std::max(o.feasibility_tolerance, o.dual_tolerance));
            }
            remove_artificials(w, o.pivot_tolerance);
        }
        auto two = iterate(w, phase_two_cost, w.original_columns, o, 2, remaining, result.telemetry,
                           result.telemetry_truncated);
        result.phase_two_iterations = two.iterations;
        result.status = two.status;
        if (two.status == SolveStatus::iteration_limit) {
            result.message = "phase II iteration limit";
            return result;
        }
        if (two.status == SolveStatus::unbounded) {
            auto anchor = full_solution(w, two.xb);
            result.primal.assign(anchor.begin(),
                                 anchor.begin() + static_cast<std::ptrdiff_t>(w.original_columns));
            result.ray.assign(two.ray.begin(),
                              two.ray.begin() + static_cast<std::ptrdiff_t>(w.original_columns));
            result.objective = dot(m.objective, result.primal) + m.objective_offset;
            result.message = "primal improving ray";
            return certify(m, std::move(result),
                           std::max(o.feasibility_tolerance, o.dual_tolerance));
        }
        if (two.status != SolveStatus::optimal) {
            result.message = "phase II numerical failure";
            return result;
        }
        auto x = full_solution(w, two.xb);
        result.primal.assign(x.begin(),
                             x.begin() + static_cast<std::ptrdiff_t>(w.original_columns));
        result.dual.assign(w.original_rows, 0);
        for (std::size_t i = 0; i < w.rows; ++i) {
            result.dual[w.row_origin[i]] = w.row_sign[i] * two.y[i];
        }
        result.basis = w.basis;
        result.objective = dot(m.objective, result.primal) + m.objective_offset;
        result.message = "reference primal revised simplex optimum";
        return certify(m, std::move(result), std::max(o.feasibility_tolerance, o.dual_tolerance));
    } catch (const std::length_error& e) {
        result.status = SolveStatus::resource_limit;
        result.message = e.what();
        return result;
    } catch (const std::exception& e) {
        result.status = SolveStatus::numerical_failure;
        result.message = e.what();
        return result;
    }
}

const char* to_string(SolveStatus s) noexcept {
    switch (s) {
    case SolveStatus::optimal:
        return "Optimal";
    case SolveStatus::infeasible:
        return "Infeasible";
    case SolveStatus::unbounded:
        return "Unbounded";
    case SolveStatus::iteration_limit:
        return "IterationLimit";
    case SolveStatus::invalid_model:
        return "InvalidModel";
    case SolveStatus::invalid_options:
        return "InvalidOptions";
    case SolveStatus::resource_limit:
        return "ResourceLimit";
    case SolveStatus::numerical_failure:
        return "NumericalFailure";
    }
    return "Unknown";
}

} // namespace markov_cero::lp::reference
