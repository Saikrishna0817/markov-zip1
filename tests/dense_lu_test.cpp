#include "markov_cero/linalg/dense_lu.hpp"
#include "markov_cero/transform/canonicalize.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
static void req(bool v, const char* m) {
    if (!v)
        throw std::runtime_error(m);
}
static bool close(double a, double b) {
    return std::abs(a - b) <= 1e-11 * std::max({1.0, std::abs(a), std::abs(b)});
}
int main() {
    using namespace markov_cero;
    linalg::DenseMatrix a{3, 3, {4, 2, 0, 2, 5, 1, 0, 1, 3}};
    std::vector<double> x{1, 2, 3};
    auto b = linalg::multiply(a, x);
    auto lu = linalg::DenseLu::factorize(a);
    auto solved = lu.solve(b);
    for (std::size_t i = 0; i < 3; ++i)
        req(close(solved[i], x[i]), "FTRAN identity");
    std::vector<double> y{3, -1, 2};
    auto bt = linalg::multiply_transpose(a, y);
    auto solved_t = lu.solve_transpose(bt);
    for (std::size_t i = 0; i < 3; ++i)
        req(close(solved_t[i], y[i]), "BTRAN identity");
    req(linalg::infinity_residual(a, solved, b) < 1e-11, "residual");
    bool bad = false;
    try {
        (void)linalg::DenseLu::factorize({2, 2, {1, 2, 2, 4}});
    } catch (const std::runtime_error&) {
        bad = true;
    }
    req(bad, "singular detection");
    std::mt19937_64 rng(0x4d324c55ULL);
    std::uniform_real_distribution<double> d(-1, 1);
    for (int trial = 0; trial < 100; ++trial) {
        linalg::DenseMatrix r{5, 5, std::vector<double>(25)};
        for (std::size_t i = 0; i < 5; ++i)
            for (std::size_t j = 0; j < 5; ++j)
                r(i, j) = d(rng);
        for (std::size_t i = 0; i < 5; ++i)
            r(i, i) += 6;
        std::vector<double> q(5);
        for (double& v : q)
            v = d(rng);
        auto rf = linalg::DenseLu::factorize(r);
        auto rb = linalg::multiply(r, q);
        auto rt = linalg::multiply_transpose(r, q);
        req(linalg::infinity_residual(r, rf.solve(rb), rb) < 1e-10, "random FTRAN");
        req(linalg::infinity_residual(r, rf.solve_transpose(rt), rt, true) < 1e-10, "random BTRAN");
    }
    model::SparseMatrixBuilder mb(2, 3);
    mb.add(0, 0, 1);
    mb.add(0, 1, 1);
    mb.add(0, 2, 1);
    mb.add(1, 0, 1);
    mb.add(1, 2, -1);
    model::Model m;
    m.name = "canonical";
    m.objective_sense = model::ObjectiveSense::maximize;
    m.matrix = mb.build();
    m.row_name = {"eq", "le"};
    m.row_lower = {model::Bound::finite(4), model::Bound::negative_infinity()};
    m.row_upper = {model::Bound::finite(4), model::Bound::finite(2)};
    m.variable_name = {"free", "fixed", "boxed"};
    m.objective = {1, 2, 3};
    m.objective_offset = 5;
    m.variable_lower = {model::Bound::negative_infinity(), model::Bound::finite(2),
                        model::Bound::finite(0)};
    m.variable_upper = {model::Bound::positive_infinity(), model::Bound::finite(2),
                        model::Bound::finite(3)};
    m.variable_type = {model::VariableType::continuous, model::VariableType::continuous,
                       model::VariableType::continuous};
    m.validate();
    auto c = transform::canonicalize(m);
    std::vector<double> z{1, 0, 1, 2, 2};
    for (double q : z)
        req(q >= 0, "canonical nonnegative");
    auto residual = linalg::infinity_residual(c.matrix, z, c.rhs);
    req(residual < 1e-12, "canonical equality residual");
    auto original = transform::reconstruct_primal(c, z);
    req(close(original[0], 1) && close(original[1], 2) && close(original[2], 1),
        "postsolve primal");
    long double cv = c.objective_offset;
    for (std::size_t j = 0; j < z.size(); ++j)
        cv += c.objective[j] * z[j];
    req(close(transform::reconstruct_objective(c, static_cast<double>(cv)), 13),
        "postsolve objective");
    linalg::DenseMatrix pivoted{3, 3, {0, 2, 1, 1, 1, 0, 2, 0, 1}};
    std::vector<double> px{1.25, -2, 0.75};
    auto plu = linalg::DenseLu::factorize(pivoted);
    auto pb = linalg::multiply(pivoted, px);
    auto ps = plu.solve(pb);
    req(linalg::infinity_residual(pivoted, ps, pb) < 1e-12, "multi-pivot FTRAN");
    auto pbt = linalg::multiply_transpose(pivoted, px);
    auto pst = plu.solve_transpose(pbt);
    req(linalg::infinity_residual(pivoted, pst, pbt, true) < 1e-12, "pivoted BTRAN");
    bad = false;
    try {
        (void)linalg::DenseLu::factorize({2, 2, {1, 0, 0, 1e-15}}, 1e-14);
    } catch (const std::runtime_error&) {
        bad = true;
    }
    req(bad, "near-singular threshold");
    bad = false;
    try {
        linalg::DenseMatrix huge{
            std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), {0}};
        huge.validate();
    } catch (const std::length_error&) {
        bad = true;
    }
    req(bad, "dense size overflow");
    bad = false;
    try {
        (void)lu.solve({std::numeric_limits<double>::infinity(), 0, 0});
    } catch (const std::overflow_error&) {
        bad = true;
    }
    req(bad, "non-finite rhs");
    bad = false;
    try {
        (void)linalg::infinity_residual({1, 1, {1}}, {std::numeric_limits<double>::quiet_NaN()},
                                        {0});
    } catch (const std::overflow_error&) {
        bad = true;
    }
    req(bad, "non-finite residual operand");
    auto discrete = m;
    discrete.variable_type[0] = model::VariableType::integer;
    bad = false;
    try {
        (void)transform::canonicalize(discrete);
    } catch (const std::invalid_argument&) {
        bad = true;
    }
    req(bad, "discrete canonicalization rejected");
    model::SparseMatrixBuilder one_mb(1, 1);
    one_mb.add(0, 0, 1);
    model::Model ranged;
    ranged.name = "ranged";
    ranged.matrix = one_mb.build();
    ranged.row_name = {"range"};
    ranged.row_lower = {model::Bound::finite(1)};
    ranged.row_upper = {model::Bound::finite(3)};
    ranged.variable_name = {"x"};
    ranged.objective = {2};
    ranged.variable_lower = {model::Bound::finite(0)};
    ranged.variable_upper = {model::Bound::positive_infinity()};
    ranged.variable_type = {model::VariableType::continuous};
    ranged.validate();
    auto rc = transform::canonicalize(ranged);
    std::vector<double> rz{2, 1, 1};
    req(linalg::infinity_residual(rc.matrix, rz, rc.rhs) < 1e-12, "ranged row slacks");
    auto rx = transform::reconstruct_primal(rc, rz);
    req(close(rx[0], 2), "ranged postsolve");
    model::SparseMatrixBuilder zero_mb(0, 2);
    model::Model one_sided;
    one_sided.name = "one-sided";
    one_sided.matrix = zero_mb.build();
    one_sided.variable_name = {"lower", "upper"};
    one_sided.objective = {1, 1};
    one_sided.variable_lower = {model::Bound::finite(2), model::Bound::negative_infinity()};
    one_sided.variable_upper = {model::Bound::positive_infinity(), model::Bound::finite(5)};
    one_sided.variable_type = {model::VariableType::continuous, model::VariableType::continuous};
    one_sided.validate();
    auto oc = transform::canonicalize(one_sided);
    auto ox = transform::reconstruct_primal(oc, {3, 4});
    req(close(ox[0], 5) && close(ox[1], 1), "one-sided reconstruction");
    auto empty = linalg::DenseLu::factorize({0, 0, {}});
    req(empty.solve({}).empty(), "empty LU");
    auto malformed = c;
    malformed.record.structural_variables = malformed.matrix.columns + 1;
    bad = false;
    try {
        malformed.validate();
    } catch (const std::invalid_argument&) {
        bad = true;
    }
    req(bad, "structural count validation");
    malformed = c;
    malformed.record.objective_sign = 2;
    bad = false;
    try {
        malformed.validate();
    } catch (const std::invalid_argument&) {
        bad = true;
    }
    req(bad, "objective sign validation");
    return 0;
}
