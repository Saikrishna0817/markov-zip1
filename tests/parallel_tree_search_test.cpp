#include "markov_cero/milp/parallel_tree_search.hpp"
#include "markov_cero/verify/primal_verifier.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

markov_cero::model::Model build_knapsack_model() {
    // max 10 x1 + 14 x2 + 12 x3
    // min -10 x1 - 14 x2 - 12 x3
    // s.t. 4 x1 + 6 x2 + 5 x3 <= 10
    // x1, x2, x3 in {0, 1}
    //
    // Continuous LP relaxation: x1=1, x3=1, x2=1/6, obj = -24.3333
    // Integer optimum: x1=1, x2=1, x3=0, obj = -24.0

    markov_cero::model::Model model;
    model.name = "KNAPSACK";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {-10.0, -14.0, -12.0};
    model.objective_offset = 0.0;

    markov_cero::model::SparseMatrixBuilder builder(1, 3);
    builder.add(0, 0, 4.0);
    builder.add(0, 1, 6.0);
    builder.add(0, 2, 5.0);
    model.matrix = builder.build();

    model.row_lower = {markov_cero::model::Bound::negative_infinity()};
    model.row_upper = {markov_cero::model::Bound::finite(10.0)};
    model.row_name = {"CAPACITY"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary
    };
    model.variable_name = {"X1", "X2", "X3"};
    model.validate();
    return model;
}

markov_cero::model::Model build_refinery_dispatch_model() {
    // MRPL Scenario: crude tanker discrete batches
    // min 50 x1 + 40 x2
    // s.t. x1 + x2 >= 3 (total batches needed >= 3)
    //      3 x1 + 2 x2 <= 8 (berth pipeline hours <= 8)
    // x1, x2 in {0, 1, 2, 3, ...}
    //
    // Valid integer points:
    // (0, 3): 3*0 + 2*3 = 6 <= 8, x1+x2 = 3 >= 3, cost = 50*0 + 40*3 = 120
    // (1, 2): 3*1 + 2*2 = 7 <= 8, x1+x2 = 3 >= 3, cost = 50*1 + 40*2 = 130
    // (2, 1): 3*2 + 2*1 = 8 <= 8, x1+x2 = 3 >= 3, cost = 50*2 + 40*1 = 140
    // Optimum is (0, 3) with cost 120

    markov_cero::model::Model model;
    model.name = "REFINERY_DISPATCH";
    model.objective_sense = markov_cero::model::ObjectiveSense::minimize;
    model.objective = {50.0, 40.0};
    model.objective_offset = 0.0;

    markov_cero::model::SparseMatrixBuilder builder(2, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(1, 0, 3.0);
    builder.add(1, 1, 2.0);
    model.matrix = builder.build();

    model.row_lower = {
        markov_cero::model::Bound::finite(3.0),
        markov_cero::model::Bound::negative_infinity()
    };
    model.row_upper = {
        markov_cero::model::Bound::positive_infinity(),
        markov_cero::model::Bound::finite(8.0)
    };
    model.row_name = {"MIN_BATCHES", "MAX_PIPELINE_HOURS"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(5.0),
        markov_cero::model::Bound::finite(5.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::integer,
        markov_cero::model::VariableType::integer
    };
    model.variable_name = {"TANKER_A", "TANKER_B"};
    model.validate();
    return model;
}

markov_cero::model::Model build_infeasible_milp_model() {
    markov_cero::model::Model model;
    model.name = "INFEASIBLE_MIP";
    model.objective = {1.0, 1.0};

    markov_cero::model::SparseMatrixBuilder builder(2, 2);
    builder.add(0, 0, 1.0);
    builder.add(0, 1, 1.0);
    builder.add(1, 0, 1.0);
    builder.add(1, 1, 1.0);
    model.matrix = builder.build();

    model.row_lower = {
        markov_cero::model::Bound::negative_infinity(),
        markov_cero::model::Bound::finite(2.0)
    };
    model.row_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::positive_infinity()
    };
    model.row_name = {"R1", "R2"};

    model.variable_lower = {
        markov_cero::model::Bound::finite(0.0),
        markov_cero::model::Bound::finite(0.0)
    };
    model.variable_upper = {
        markov_cero::model::Bound::finite(1.0),
        markov_cero::model::Bound::finite(1.0)
    };
    model.variable_type = {
        markov_cero::model::VariableType::binary,
        markov_cero::model::VariableType::binary
    };
    model.variable_name = {"X1", "X2"};
    model.validate();
    return model;
}

void test_knapsack_multi_threads() {
    const auto model = build_knapsack_model();

    // 1-thread
    markov_cero::milp::ParallelOptions opt1;
    opt1.num_threads = 1;
    const auto res1 = markov_cero::milp::solve_parallel(model, opt1);
    assert(res1.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res1.objective - (-24.0)) < 1e-5);

    // 2-threads
    markov_cero::milp::ParallelOptions opt2;
    opt2.num_threads = 2;
    const auto res2 = markov_cero::milp::solve_parallel(model, opt2);
    assert(res2.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res2.objective - (-24.0)) < 1e-5);

    // 4-threads
    markov_cero::milp::ParallelOptions opt4;
    opt4.num_threads = 4;
    const auto res4 = markov_cero::milp::solve_parallel(model, opt4);
    assert(res4.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res4.objective - (-24.0)) < 1e-5);

    // Confirm identical optimal integer objective across threads
    assert(std::abs(res1.objective - res2.objective) < 1e-6);
    assert(std::abs(res1.objective - res4.objective) < 1e-6);

    // Verify primal feasibility and integrality
    for (const auto& res : {res1, res2, res4}) {
        markov_cero::verify::Candidate candidate{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(model, candidate);
        assert(report.passed);
        assert(report.maximum_integrality_violation < 1e-6);
        assert(std::abs(res.primal[0] - 1.0) < 1e-5);
        assert(std::abs(res.primal[1] - 1.0) < 1e-5);
        assert(std::abs(res.primal[2] - 0.0) < 1e-5);
    }

    std::cout << "[+] test_knapsack_multi_threads passed (1, 2, 4 threads verified)\n";
}

void test_refinery_dispatch_multi_threads() {
    const auto model = build_refinery_dispatch_model();

    // 1-thread
    markov_cero::milp::ParallelOptions opt1;
    opt1.num_threads = 1;
    const auto res1 = markov_cero::milp::solve_parallel(model, opt1);
    assert(res1.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res1.objective - 120.0) < 1e-5);

    // 2-threads
    markov_cero::milp::ParallelOptions opt2;
    opt2.num_threads = 2;
    const auto res2 = markov_cero::milp::solve_parallel(model, opt2);
    assert(res2.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res2.objective - 120.0) < 1e-5);

    // 4-threads
    markov_cero::milp::ParallelOptions opt4;
    opt4.num_threads = 4;
    const auto res4 = markov_cero::milp::solve_parallel(model, opt4);
    assert(res4.status == markov_cero::lp::reference::SolveStatus::optimal);
    assert(std::abs(res4.objective - 120.0) < 1e-5);

    // Confirm identical optimal integer objective across threads
    assert(std::abs(res1.objective - res2.objective) < 1e-6);
    assert(std::abs(res1.objective - res4.objective) < 1e-6);

    // Verify primal feasibility and integrality
    for (const auto& res : {res1, res2, res4}) {
        markov_cero::verify::Candidate candidate{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(model, candidate);
        assert(report.passed);
        assert(report.maximum_integrality_violation < 1e-6);
        assert(std::abs(res.primal[0] - 0.0) < 1e-5);
        assert(std::abs(res.primal[1] - 3.0) < 1e-5);
    }

    std::cout << "[+] test_refinery_dispatch_multi_threads passed (1, 2, 4 threads verified)\n";
}

void test_thread_safety_repeated_runs() {
    const auto knapsack = build_knapsack_model();
    const auto refinery = build_refinery_dispatch_model();

    markov_cero::milp::ParallelOptions opt;
    opt.num_threads = 4;

    // Run 20 iterations back-to-back on knapsack
    for (std::size_t i = 0; i < 20; ++i) {
        const auto res = markov_cero::milp::solve_parallel(knapsack, opt);
        assert(res.status == markov_cero::lp::reference::SolveStatus::optimal);
        assert(std::abs(res.objective - (-24.0)) < 1e-5);

        markov_cero::verify::Candidate candidate{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(knapsack, candidate);
        assert(report.passed);
        assert(report.maximum_integrality_violation < 1e-6);
    }

    // Run 20 iterations back-to-back on refinery dispatch
    for (std::size_t i = 0; i < 20; ++i) {
        const auto res = markov_cero::milp::solve_parallel(refinery, opt);
        assert(res.status == markov_cero::lp::reference::SolveStatus::optimal);
        assert(std::abs(res.objective - 120.0) < 1e-5);

        markov_cero::verify::Candidate candidate{res.primal, res.objective};
        const auto report = markov_cero::verify::verify_primal(refinery, candidate);
        assert(report.passed);
        assert(report.maximum_integrality_violation < 1e-6);
    }

    std::cout << "[+] test_thread_safety_repeated_runs passed (40 stress runs completed without race or deadlock)\n";
}

void test_infeasible_parallel() {
    const auto model = build_infeasible_milp_model();

    for (std::size_t th : {1, 2, 4}) {
        markov_cero::milp::ParallelOptions opt;
        opt.num_threads = th;
        const auto res = markov_cero::milp::solve_parallel(model, opt);
        assert(res.status == markov_cero::lp::reference::SolveStatus::infeasible);
    }

    std::cout << "[+] test_infeasible_parallel passed\n";
}

void test_thread_safe_queue_unit() {
    markov_cero::milp::ThreadSafeNodeQueue queue;

    auto n1 = std::make_shared<markov_cero::milp::BranchNode>();
    n1->id = 1;
    n1->lower_bound = 10.0;

    auto n2 = std::make_shared<markov_cero::milp::BranchNode>();
    n2->id = 2;
    n2->lower_bound = 5.0;

    auto n3 = std::make_shared<markov_cero::milp::BranchNode>();
    n3->id = 3;
    n3->lower_bound = 20.0;

    queue.push(n1);
    queue.push(n2);
    queue.push(n3);

    assert(queue.size() == 3);
    assert(std::abs(queue.min_lower_bound() - 5.0) < 1e-9);

    // Prune nodes with lower_bound >= 15.0 (should prune n3)
    queue.prune(15.0);
    assert(queue.size() == 2);

    bool became_active = false;
    auto popped1 = queue.pop_node(false, 100.0, became_active);
    assert(popped1 != nullptr);
    assert(popped1->id == 2); // lowest lower bound popped first
    assert(became_active);

    auto popped2 = queue.pop_node(true, 100.0, became_active);
    assert(popped2 != nullptr);
    assert(popped2->id == 1);
    assert(became_active);

    queue.deactivate_worker();
    assert(queue.empty());

    std::cout << "[+] test_thread_safe_queue_unit passed\n";
}

void test_incumbent_manager_unit() {
    markov_cero::milp::IncumbentManager mgr(100.0);
    assert(mgr.has_incumbent());
    assert(std::abs(mgr.get_objective() - 100.0) < 1e-9);

    // Try worsening update: should fail
    bool up1 = mgr.update_if_better(120.0, {1.0, 2.0});
    assert(!up1);
    assert(std::abs(mgr.get_objective() - 100.0) < 1e-9);

    // Try improving update: should succeed
    bool up2 = mgr.update_if_better(80.0, {3.0, 4.0});
    assert(up2);
    assert(std::abs(mgr.get_objective() - 80.0) < 1e-9);
    const auto p = mgr.get_primal();
    assert(p.size() == 2);
    assert(p[0] == 3.0 && p[1] == 4.0);

    // Concurrent updates test
    std::vector<std::jthread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&mgr, i]() {
            double obj = 70.0 - static_cast<double>(i);
            mgr.update_if_better(obj, {obj, obj * 2.0});
        });
    }
    threads.clear();

    assert(mgr.get_objective() <= 63.0 + 1e-9);

    std::cout << "[+] test_incumbent_manager_unit passed\n";
}

} // namespace

int main() {
    try {
        test_thread_safe_queue_unit();
        test_incumbent_manager_unit();
        test_knapsack_multi_threads();
        test_refinery_dispatch_multi_threads();
        test_thread_safety_repeated_runs();
        test_infeasible_parallel();
        std::cout << "\n=======================================================\n";
        std::cout << "All Parallel Tree Search Tests PASSED Successfully!\n";
        std::cout << "=======================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
