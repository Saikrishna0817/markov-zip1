#include "markov_cero/io/mps.hpp"
#include "markov_cero/milp/milp_solver.hpp"
#include "markov_cero/qp/admm_solver.hpp"
#include "markov_cero/qp/kkt.hpp"
#include "markov_cero/qp/model.hpp"
#include "markov_cero/qp/verifier.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    using namespace markov_cero;
    using namespace markov_cero::qp;

    std::cout << "[Test] 1. KktSolver LDL^T factorization and solve\n";
    {
        SparseSymmetricMatrix P;
        P.dimension = 2;
        P.column_offsets = {0, 1, 3};
        P.row_indices = {0, 0, 1};
        P.values = {4.0, 1.0, 2.0};

        // A = [1 1] (1 x 2)
        linalg::SparseCsc A;
        A.rows = 1;
        A.columns = 2;
        A.column_offsets = {0, 1, 2};
        A.row_indices = {0, 0};
        A.values = {1.0, 1.0};

        const double sigma = 1e-4;
        const std::vector<double> rho = {1.0};

        KktSolver kkt;
        require(kkt.factorize(P, A, sigma, rho), "KKT factorization succeeded");
        require(kkt.dimension() == 3, "KKT dimension is 3");

        std::vector<double> sol_x, sol_nu;
        kkt.solve({1.0, 2.0}, {3.0}, sol_x, sol_nu);
        require(sol_x.size() == 2, "sol_x size");
        require(sol_nu.size() == 1, "sol_nu size");
    }

    std::cout << "[Test] 2. Unconstrained Convex QP\n";
    // min (1/2)(x1^2 + x2^2) - x1 - 2*x2
    // Analytic minimum: x1* = 1, x2* = 2, f(x*) = 0.5*(1+4) - 1 - 4 = -2.5
    {
        const std::string mps =
            "NAME UNCON\n"
            "ROWS\n"
            " N OBJ\n"
            "COLUMNS\n"
            " X1 OBJ -1\n"
            " X2 OBJ -2\n"
            "QUADOBJ\n"
            " X1 X1 1\n"
            " X2 X2 1\n"
            "BOUNDS\n"
            " FR B X1\n"
            " FR B X2\n"
            "ENDATA\n";

        const auto model = io::parse_mps_string(mps);
        const auto qp = make_quadratic_model(model);
        require(check_convexity(qp.P), "model is convex");

        QpOptions opts;
        opts.absolute_tolerance = 1e-5;
        opts.relative_tolerance = 1e-5;
        const auto sol = solve_qp(qp, opts);
        require(sol.status == QpStatus::optimal, "unconstrained QP optimal");
        require(std::abs(sol.x[0] - 1.0) < 1e-3, "x1 near 1.0");
        require(std::abs(sol.x[1] - 2.0) < 1e-3, "x2 near 2.0");
        require(std::abs(sol.objective_value - (-2.5)) < 1e-3, "objective near -2.5");

        const auto rep = verify_qp_solution(qp, sol);
        require(rep.passed, "verifier passes on unconstrained QP");
    }

    std::cout << "[Test] 3. Constrained Convex QP (Equality & Inequality)\n";
    // min (1/2)(4*x1^2 + 2*x1*x2 + 2*x2^2) + x1 + 2*x2
    // s.t. x1 + x2 >= 1, x1 >= 0, x2 >= 0
    {
        const std::string mps =
            "NAME CONSTR\n"
            "ROWS\n"
            " N OBJ\n"
            " G C1\n"
            "COLUMNS\n"
            " X1 OBJ 1 C1 1\n"
            " X2 OBJ 2 C1 1\n"
            "RHS\n"
            " RHS1 C1 1\n"
            "QUADOBJ\n"
            " X1 X1 4\n"
            " X1 X2 1\n"
            " X2 X2 2\n"
            "ENDATA\n";

        const auto model = io::parse_mps_string(mps);
        const auto qp = make_quadratic_model(model);

        QpOptions opts;
        opts.absolute_tolerance = 1e-5;
        opts.relative_tolerance = 1e-5;
        const auto sol = solve_qp(qp, opts);
        require(sol.status == QpStatus::optimal, "constrained QP optimal");
        require(sol.x[0] + sol.x[1] >= 1.0 - 1e-4, "constraint satisfied");

        const auto rep = verify_qp_solution(qp, sol);
        require(rep.passed, "verifier passes on constrained QP");
    }

    std::cout << "[Test] 4. Non-Convex QP Rejection\n";
    // P has negative eigenvalue -> non-convex
    {
        SparseSymmetricMatrix P;
        P.dimension = 2;
        P.column_offsets = {0, 1, 2};
        P.row_indices = {0, 1};
        P.values = {-1.0, 2.0};

        require(!check_convexity(P), "non-convex detected by check_convexity");

        QuadraticModel qp;
        qp.P = P;
        qp.q = {1.0, 1.0};
        qp.A.rows = 0;
        qp.A.columns = 2;
        qp.A.column_offsets = {0, 0, 0};

        const auto sol = solve_qp(qp);
        require(sol.status == QpStatus::non_convex, "solver safely rejected non-convex QP");
    }

    std::cout << "[Test] 5. Primal Infeasible QP\n";
    // x1 + x2 >= 5 and x1 + x2 <= 2
    {
        const std::string mps =
            "NAME INFEAS\n"
            "ROWS\n"
            " N OBJ\n"
            " G C1\n"
            " L C2\n"
            "COLUMNS\n"
            " X1 OBJ 1 C1 1\n"
            " X1 C2 1\n"
            " X2 OBJ 1 C1 1\n"
            " X2 C2 1\n"
            "RHS\n"
            " RHS1 C1 5\n"
            " RHS1 C2 2\n"
            "QUADOBJ\n"
            " X1 X1 2\n"
            " X2 X2 2\n"
            "ENDATA\n";

        const auto model = io::parse_mps_string(mps);
        const auto qp = make_quadratic_model(model);

        const auto sol = solve_qp(qp);
        require(sol.status == QpStatus::primal_infeasible, "detected primal infeasibility");
    }

    std::cout << "[Test] 6. Maros-Meszaros Style Benchmark Form (HUEBNER style)\n";
    // min (1/2) sum_i (x_i - i)^2 s.t. sum_i x_i = N*(N+1)/2, x >= 0
    {
        const std::size_t N = 10;
        model::Model m;
        m.name = "MM_SYNTH";
        m.objective_sense = model::ObjectiveSense::minimize;
        m.matrix = model::SparseMatrixBuilder(1, N).build();
        for (std::size_t j = 0; j < N; ++j) {
            m.variable_name.push_back("x" + std::to_string(j));
            m.objective.push_back(-static_cast<double>(j + 1));
            m.variable_lower.push_back(model::Bound::finite(0.0));
            m.variable_upper.push_back(model::Bound::positive_infinity());
            m.variable_type.push_back(model::VariableType::continuous);
        }
        m.row_name = {"eq1"};
        const double target_sum = static_cast<double>(N * (N + 1) / 2);
        m.row_lower = {model::Bound::finite(target_sum)};
        m.row_upper = {model::Bound::finite(target_sum)};

        model::SparseMatrixBuilder ab(1, N);
        for (std::size_t j = 0; j < N; ++j) {
            ab.add(0, j, 1.0);
        }
        m.matrix = ab.build();

        model::SparseMatrixBuilder qb(N, N);
        for (std::size_t j = 0; j < N; ++j) {
            qb.add(j, j, 1.0);
        }
        m.has_quadratic_objective = true;
        m.quadratic_matrix = qb.build();
        m.validate();

        const auto qp = make_quadratic_model(m);
        QpOptions opts;
        opts.absolute_tolerance = 1e-5;
        opts.relative_tolerance = 1e-5;
        opts.max_iterations = 2000;
        const auto sol = solve_qp(qp, opts);
        require(sol.status == QpStatus::optimal, "MM benchmark optimal");
        for (std::size_t j = 0; j < N; ++j) {
            require(std::abs(sol.x[j] - static_cast<double>(j + 1)) < 1e-2,
                    "MM benchmark variable matches analytic target");
        }
        const auto rep = verify_qp_solution(qp, sol);
        require(rep.passed, "verifier passes on MM benchmark");
    }

    std::cout << "[Test] 7. Mixed-Integer Quadratic Program (MIQP)\n";
    // min (1/2)(2*x1^2 + 2*x2^2) - 2.4*x1 - 5.6*x2 + 9.28
    // s.t. x1 + x2 <= 3, x1, x2 in {0, 1, 2, 3} integers
    // Integer optimum: x* = (1, 2), f(x*) = 0.68
    {
        model::Model miqp;
        miqp.name = "MIQP_TOY";
        miqp.objective_sense = model::ObjectiveSense::minimize;
        miqp.objective_offset = 9.28;
        miqp.variable_name = {"x1", "x2"};
        miqp.objective = {-2.4, -5.6};
        miqp.variable_lower = {model::Bound::finite(0.0), model::Bound::finite(0.0)};
        miqp.variable_upper = {model::Bound::finite(3.0), model::Bound::finite(3.0)};
        miqp.variable_type = {model::VariableType::integer, model::VariableType::integer};

        miqp.row_name = {"c1"};
        miqp.row_lower = {model::Bound::negative_infinity()};
        miqp.row_upper = {model::Bound::finite(3.0)};

        model::SparseMatrixBuilder ab(1, 2);
        ab.add(0, 0, 1.0);
        ab.add(0, 1, 1.0);
        miqp.matrix = ab.build();

        model::SparseMatrixBuilder qb(2, 2);
        qb.add(0, 0, 2.0);
        qb.add(1, 1, 2.0);
        miqp.has_quadratic_objective = true;
        miqp.quadratic_matrix = qb.build();
        miqp.validate();

        milp::Options opts;
        opts.max_iterations = 2000;
        opts.feasibility_tolerance = 1e-4;
        opts.integrality_tolerance = 1e-4;
        const auto res = milp::solve(miqp, opts);
        require(res.status == lp::reference::SolveStatus::optimal,
                "MIQP solved to optimality");
        require(std::round(res.primal[0]) == 1.0, "x1 is 1");
        require(std::round(res.primal[1]) == 2.0, "x2 is 2");
        require(std::abs(res.objective - 0.68) < 1e-2, "MIQP objective is ~0.68");
    }

    std::cout << "[Pass] All QP tests passed successfully!\n";
    return 0;
}
