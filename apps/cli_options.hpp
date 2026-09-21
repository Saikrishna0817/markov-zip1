#pragma once

#include "markov_cero/lp/reference/revised_simplex.hpp"
#include "markov_cero/milp/milp_solver.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>

namespace markov_cero::apps {

struct CliOptions {
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

    lp::reference::Options options;
    milp::Options milp_options;

    bool help_requested = false;
    bool error = false;
    int exit_code = 0;

    static void usage(std::ostream& out) {
        out << "usage: markov-cero-solve MODEL.mps [options]\n"
            << "options:\n"
            << "  --output result.json     Write output JSON to file\n"
            << "  --engine primal|dual|pdlp|milp|parallel|auto "
            << "Select solver engine (default: auto)\n"
            << "  --threads N              Worker threads for parallel tree search (default: 4)\n"
            << "  --branching most_fractional|pseudo_cost|strong_branching|reliability Branching "
               "variable selection rule (default: pseudo_cost)\n"
            << "  --iteration-limit N      Maximum simplex iterations\n"
            << "  --max-nodes N            Maximum branch-and-cut search nodes (default: 50000)\n"
            << "  --time-limit SEC         Maximum search time limit in seconds (default: 60.0)\n"
            << "  --cuts, --no-cuts        Enable or disable Gomory & MIR mixed-integer cuts "
               "(default: enabled)\n"
            << "  --heuristics, --no-heuristics Enable or disable primal heuristics (default: "
               "enabled)\n"
            << "  --warm-start FILE        Load warm-start basis from file (dual engine)\n"
            << "  --save-basis FILE        Save optimal basis to file\n"
            << "  --presolve, --no-presolve Enable or disable presolve reductions (default: "
               "enabled)\n"
            << "  --scale, --no-scale       Enable or disable Ruiz matrix scaling (default: "
               "enabled)\n"
            << "  --max-presolve-passes N   Maximum presolve passes (default: 5)\n"
            << "  --ruiz-iterations N       Maximum Ruiz equilibration iterations (default: 10)\n"
            << "  --tolerance TOL          Relative KKT tolerance for PDLP (default: 1e-4)\n"
            << "  --backend cpu|gpu        PDLP execution backend (default: cpu)\n"
            << "  --help, -h               Show this help\n";
    }

    static CliOptions parse(int argc, char** argv) {
        CliOptions parsed;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(std::cout);
                parsed.help_requested = true;
                return parsed;
            }
            if (arg == "--output") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.output_path = argv[++i];
                continue;
            }
            if (arg == "--engine") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.engine_name = argv[++i];
                if (parsed.engine_name != "primal" && parsed.engine_name != "dual" &&
                    parsed.engine_name != "pdlp" && parsed.engine_name != "milp" &&
                    parsed.engine_name != "parallel" && parsed.engine_name != "qp" &&
                    parsed.engine_name != "miqp" && parsed.engine_name != "auto") {
                    std::cerr << "invalid engine: " << parsed.engine_name << "\n";
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                continue;
            }
            if (arg == "--threads") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.num_threads = std::strtoul(argv[++i], nullptr, 10);
                if (parsed.num_threads == 0) parsed.num_threads = 1;
                continue;
            }
            if (arg == "--branching") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                const std::string bval = argv[++i];
                if (bval == "most_fractional") {
                    parsed.milp_options.branching_strategy =
                        milp::BranchingStrategy::most_fractional;
                } else if (bval == "pseudo_cost") {
                    parsed.milp_options.branching_strategy = milp::BranchingStrategy::pseudo_cost;
                } else if (bval == "strong_branching") {
                    parsed.milp_options.branching_strategy =
                        milp::BranchingStrategy::strong_branching;
                } else if (bval == "reliability") {
                    parsed.milp_options.branching_strategy = milp::BranchingStrategy::reliability;
                } else {
                    std::cerr << "invalid branching strategy: " << bval << "\n";
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                continue;
            }
            if (arg == "--max-nodes") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.milp_options.max_nodes = std::strtoul(argv[++i], nullptr, 10);
                continue;
            }
            if (arg == "--time-limit") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.milp_options.time_limit_seconds = std::strtod(argv[++i], nullptr);
                continue;
            }
            if (arg == "--cuts") {
                parsed.milp_options.enable_cuts = true;
                continue;
            }
            if (arg == "--no-cuts") {
                parsed.milp_options.enable_cuts = false;
                continue;
            }
            if (arg == "--heuristics") {
                parsed.milp_options.enable_heuristics = true;
                continue;
            }
            if (arg == "--no-heuristics") {
                parsed.milp_options.enable_heuristics = false;
                continue;
            }
            if (arg == "--warm-start") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.warm_start_path = argv[++i];
                continue;
            }
            if (arg == "--save-basis") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.save_basis_path = argv[++i];
                continue;
            }
            if (arg == "--presolve") {
                parsed.enable_presolve = true;
                continue;
            }
            if (arg == "--no-presolve") {
                parsed.enable_presolve = false;
                continue;
            }
            if (arg == "--scale") {
                parsed.enable_scale = true;
                continue;
            }
            if (arg == "--no-scale") {
                parsed.enable_scale = false;
                continue;
            }
            if (arg == "--max-presolve-passes") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.max_presolve_passes = std::strtoul(argv[++i], nullptr, 10);
                continue;
            }
            if (arg == "--ruiz-iterations") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.ruiz_iterations = std::strtoul(argv[++i], nullptr, 10);
                continue;
            }
            if (arg == "--tolerance") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.pdlp_tolerance = std::strtod(argv[++i], nullptr);
                if (parsed.pdlp_tolerance <= 0.0) {
                    std::cerr << "tolerance must be positive: " << parsed.pdlp_tolerance << "\n";
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                continue;
            }
            if (arg == "--backend") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.backend_name = argv[++i];
                if (parsed.backend_name != "cpu" && parsed.backend_name != "gpu") {
                    std::cerr << "invalid backend: " << parsed.backend_name << "\n";
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                continue;
            }
            if (arg == "--iteration-limit") {
                if (i + 1 >= argc) {
                    usage(std::cerr);
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                char* endptr = nullptr;
                errno = 0;
                const unsigned long long val = std::strtoull(argv[++i], &endptr, 10);
                if (errno == ERANGE || endptr == argv[i] || *endptr != '\0') {
                    std::cerr << "invalid iteration limit: " << argv[i] << "\n";
                    parsed.error = true; parsed.exit_code = 8; return parsed;
                }
                parsed.options.iteration_limit = static_cast<std::size_t>(val);
                parsed.milp_options.max_iterations = static_cast<std::size_t>(val);
                continue;
            }
            if (!arg.empty() && arg[0] == '-') {
                usage(std::cerr);
                parsed.error = true; parsed.exit_code = 8; return parsed;
            }
            if (!parsed.path.empty()) {
                usage(std::cerr);
                parsed.error = true; parsed.exit_code = 8; return parsed;
            }
            parsed.path = arg;
        }
        if (parsed.path.empty()) {
            usage(std::cerr);
            parsed.error = true; parsed.exit_code = 8; return parsed;
        }
        return parsed;
    }
};

} // namespace markov_cero::apps
