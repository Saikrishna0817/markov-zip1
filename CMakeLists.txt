cmake_minimum_required(VERSION 3.25)
project(markov-cero VERSION 0.5.2 LANGUAGES CXX)
option(MARKOV_CERO_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(MARKOV_CERO_ENABLE_ASAN_UBSAN "Enable address/undefined sanitizers" OFF)
option(MARKOV_CERO_ENABLE_TSAN "Enable thread sanitizer" OFF)
option(MARKOV_CERO_BUILD_FUZZER "Build libFuzzer MPS target with Clang" OFF)
option(MARKOV_CERO_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
if(MARKOV_CERO_ENABLE_ASAN_UBSAN AND MARKOV_CERO_ENABLE_TSAN)
  message(FATAL_ERROR "ASan/UBSan and TSan are mutually exclusive")
endif()
if(MARKOV_CERO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")
    endif()
    message(STATUS
      "CUDA enabled: compiler=${CMAKE_CUDA_COMPILER}, arch=${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING
      "MARKOV_CERO_ENABLE_CUDA requested, but no CUDA compiler found. Graceful fallback to CPU.")
    set(MARKOV_CERO_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA GPU acceleration" FORCE)
  endif()
endif()
find_package(Threads REQUIRED)
add_library(markov_cero_core STATIC
  src/foundation/build_info.cpp
  src/model/model.cpp
  src/io/mps.cpp
  src/verify/primal_verifier.cpp
  src/linalg/dense_lu.cpp
  src/linalg/sparse_basis.cpp
  src/transform/canonicalize.cpp
  src/transform/sparse_canonicalize.cpp
  src/presolve/presolve.cpp
  src/scale/ruiz_scaling.cpp
  src/lp/reference/revised_simplex.cpp
  src/verify/reference_lp_verifier.cpp
  src/lp/dual/dual_simplex.cpp
  src/lp/first_order/pdlp.cpp
  src/milp/branch_selector.cpp
  src/milp/cut_pool.cpp
  src/milp/gomory.cpp
  src/milp/heuristics.cpp
  src/milp/milp_solver.cpp
  src/milp/node_lp.cpp
  src/milp/mir.cpp
  src/milp/parallel_tree_search.cpp
  src/milp/shared_incumbent.cpp
  src/milp/strong_branching.cpp
  src/milp/work_queue.cpp
  src/qp/model.cpp
  src/qp/kkt.cpp
  src/qp/admm_solver.cpp
  src/qp/verifier.cpp
  gpu/src/buffer.cpp
  gpu/src/csr.cpp
  gpu/src/device.cpp
  gpu/src/pdhg_step.cpp
  gpu/src/reduce.cpp
  gpu/src/spmv.cpp
  gpu/src/vector_ops.cpp
)
target_include_directories(markov_cero_core PUBLIC include gpu/include)
target_compile_features(markov_cero_core PUBLIC cxx_std_20)
target_link_libraries(markov_cero_core PUBLIC Threads::Threads)
if(MARKOV_CERO_ENABLE_CUDA)
  target_compile_definitions(markov_cero_core PUBLIC MARKOV_CERO_HAS_CUDA=1)
  target_sources(markov_cero_core PRIVATE
    gpu/kernels/pdhg_step.cu
    gpu/kernels/reduce.cu
    gpu/kernels/spmv.cu
    gpu/kernels/vector_ops.cu
  )
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(markov_cero_core PRIVATE -Wall -Wextra -Wpedantic)
  if(MARKOV_CERO_WARNINGS_AS_ERRORS)
    target_compile_options(markov_cero_core PRIVATE -Werror)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(markov_cero_core PRIVATE -Wno-maybe-uninitialized)
  endif()
  if(MARKOV_CERO_ENABLE_ASAN_UBSAN)
    target_compile_options(markov_cero_core PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(markov_cero_core PUBLIC -fsanitize=address,undefined)
  elseif(MARKOV_CERO_ENABLE_TSAN)
    target_compile_options(markov_cero_core PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(markov_cero_core PUBLIC -fsanitize=thread)
  endif()
endif()
add_executable(markov-cero-info apps/markov_cero_info.cpp)
target_link_libraries(markov-cero-info PRIVATE markov_cero_core)
add_executable(markov-cero-mps-inspect apps/markov_cero_mps_inspect.cpp)
target_link_libraries(markov-cero-mps-inspect PRIVATE markov_cero_core)
add_executable(markov-cero-solve apps/markov_cero_solve.cpp)
target_link_libraries(markov-cero-solve PRIVATE markov_cero_core)
enable_testing()
add_executable(build_info_test tests/build_info_test.cpp)
target_link_libraries(build_info_test PRIVATE markov_cero_core)
add_executable(model_test tests/model_test.cpp)
target_link_libraries(model_test PRIVATE markov_cero_core)
add_executable(mps_parser_test tests/mps_parser_test.cpp)
target_link_libraries(mps_parser_test PRIVATE markov_cero_core)
add_executable(mps_fuzz_smoke tests/fuzz/mps_fuzz_smoke.cpp)
target_link_libraries(mps_fuzz_smoke PRIVATE markov_cero_core)
add_executable(model_property_test tests/model_property_test.cpp)
target_link_libraries(model_property_test PRIVATE markov_cero_core)
add_executable(dense_lu_test tests/dense_lu_test.cpp)
target_link_libraries(dense_lu_test PRIVATE markov_cero_core)
add_executable(primal_simplex_test tests/primal_simplex_test.cpp)
target_link_libraries(primal_simplex_test PRIVATE markov_cero_core)
add_executable(primal_simplex_property_test tests/primal_simplex_property_test.cpp)
target_link_libraries(primal_simplex_property_test PRIVATE markov_cero_core)
add_executable(dual_simplex_test tests/dual_simplex_test.cpp)
target_link_libraries(dual_simplex_test PRIVATE markov_cero_core)
add_executable(warm_start_property_test tests/warm_start_property_test.cpp)
target_link_libraries(warm_start_property_test PRIVATE markov_cero_core)
add_executable(sparse_basis_test tests/sparse_basis_test.cpp)
target_link_libraries(sparse_basis_test PRIVATE markov_cero_core)
add_executable(sparse_update_property_test tests/sparse_update_property_test.cpp)
target_link_libraries(sparse_update_property_test PRIVATE markov_cero_core)
add_executable(regression_test tests/regression_test.cpp)
target_link_libraries(regression_test PRIVATE markov_cero_core)
add_executable(sparse_canonicalize_test tests/sparse_canonicalize_test.cpp)
target_link_libraries(sparse_canonicalize_test PRIVATE markov_cero_core)
add_executable(presolve_test tests/presolve_test.cpp)
target_link_libraries(presolve_test PRIVATE markov_cero_core)
add_executable(ruiz_scaling_test tests/ruiz_scaling_test.cpp)
target_link_libraries(ruiz_scaling_test PRIVATE markov_cero_core)
add_executable(milp_test tests/milp_test.cpp)
target_link_libraries(milp_test PRIVATE markov_cero_core)
add_executable(milp_heuristics_test tests/milp_heuristics_test.cpp)
target_link_libraries(milp_heuristics_test PRIVATE markov_cero_core)
add_executable(milp_cuts_test tests/milp_cuts_test.cpp)
target_link_libraries(milp_cuts_test PRIVATE markov_cero_core)
add_executable(strong_branching_test tests/strong_branching_test.cpp)
target_link_libraries(strong_branching_test PRIVATE markov_cero_core)
add_executable(pdlp_test tests/pdlp_test.cpp)
target_link_libraries(pdlp_test PRIVATE markov_cero_core)
add_executable(parallel_tree_search_test tests/parallel_tree_search_test.cpp)
target_link_libraries(parallel_tree_search_test PRIVATE markov_cero_core)
add_executable(gpu_buffer_test gpu/tests/gpu_buffer_test.cpp)
target_link_libraries(gpu_buffer_test PRIVATE markov_cero_core)
add_executable(equivalence_test gpu/tests/equivalence_test.cpp)
target_link_libraries(equivalence_test PRIVATE markov_cero_core)
add_executable(gpu_reduction_test gpu/tests/reduction_test.cpp)
target_link_libraries(gpu_reduction_test PRIVATE markov_cero_core)
add_executable(gpu_pdhg_step_test gpu/tests/pdhg_step_test.cpp)
target_link_libraries(gpu_pdhg_step_test PRIVATE markov_cero_core)
add_executable(gpu_pdhg_restart_test gpu/tests/pdhg_restart_test.cpp)
target_link_libraries(gpu_pdhg_restart_test PRIVATE markov_cero_core)
add_executable(gpu_pdhg_adaptive_test gpu/tests/pdhg_adaptive_test.cpp)
target_link_libraries(gpu_pdhg_adaptive_test PRIVATE markov_cero_core)
add_executable(gpu_pdhg_kkt_test gpu/tests/pdhg_kkt_test.cpp)
target_link_libraries(gpu_pdhg_kkt_test PRIVATE markov_cero_core)
add_executable(gpu_pdhg_timing_test gpu/tests/pdhg_timing_test.cpp)
target_link_libraries(gpu_pdhg_timing_test PRIVATE markov_cero_core)
add_executable(qp_test tests/qp_test.cpp)
target_link_libraries(qp_test PRIVATE markov_cero_core)
add_test(NAME build_info COMMAND build_info_test)
add_test(NAME model_verifier COMMAND model_test)
add_test(NAME mps_parser COMMAND mps_parser_test)
add_test(NAME mps_fuzz_smoke COMMAND mps_fuzz_smoke)
add_test(NAME model_properties COMMAND model_property_test)
add_test(NAME dense_lu COMMAND dense_lu_test)
add_test(NAME primal_simplex COMMAND primal_simplex_test)
add_test(NAME primal_simplex_properties COMMAND primal_simplex_property_test)
add_test(NAME dual_simplex COMMAND dual_simplex_test)
add_test(NAME warm_start_properties COMMAND warm_start_property_test)
add_test(NAME sparse_basis COMMAND sparse_basis_test)
add_test(NAME sparse_update_properties COMMAND sparse_update_property_test)
add_test(NAME audit_regressions COMMAND regression_test)
add_test(NAME sparse_canonicalize COMMAND sparse_canonicalize_test)
add_test(NAME presolve COMMAND presolve_test)
add_test(NAME ruiz_scaling COMMAND ruiz_scaling_test)
add_test(NAME milp COMMAND milp_test)
add_test(NAME milp_heuristics COMMAND milp_heuristics_test)
add_test(NAME milp_cuts COMMAND milp_cuts_test)
add_test(NAME strong_branching COMMAND strong_branching_test)
add_test(NAME pdlp COMMAND pdlp_test)
add_test(NAME parallel_tree_search COMMAND parallel_tree_search_test)
add_test(NAME gpu_buffer COMMAND gpu_buffer_test)
add_test(NAME equivalence COMMAND equivalence_test)
add_test(NAME gpu_reduction COMMAND gpu_reduction_test)
add_test(NAME gpu_pdhg_step COMMAND gpu_pdhg_step_test)
add_test(NAME gpu_pdhg_restart COMMAND gpu_pdhg_restart_test)
add_test(NAME gpu_pdhg_adaptive COMMAND gpu_pdhg_adaptive_test)
add_test(NAME gpu_pdhg_kkt COMMAND gpu_pdhg_kkt_test)
add_test(NAME gpu_pdhg_timing COMMAND gpu_pdhg_timing_test)
add_test(NAME qp COMMAND qp_test)
add_test(NAME json_records COMMAND python3 -c "import json, glob; [json.load(open(f)) for f in glob.glob('**/*.json', recursive=True) if not f.startswith('_')]")
add_test(NAME sovereignty_guard
  COMMAND ${CMAKE_COMMAND} -E env python3
    ${CMAKE_SOURCE_DIR}/scripts/check-sovereignty.py
    ${CMAKE_SOURCE_DIR}
    --binary $<TARGET_FILE:markov-cero-solve>)
add_test(NAME cli_blend_optimal COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/blend.mps)
add_test(NAME cli_qp_portfolio
  COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/qp_portfolio.mps)
add_test(NAME cli_refinery_feasible COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/refinery/refinery-feasible.mps)
add_test(NAME cli_refinery_infeasible COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/refinery/refinery-infeasible.mps)
set_tests_properties(cli_refinery_infeasible PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_refinery_malformed COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/refinery/refinery-malformed.mps)
set_tests_properties(cli_refinery_malformed PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_refinery_limited COMMAND markov-cero-solve ${CMAKE_SOURCE_DIR}/examples/refinery/refinery-limited.mps --iteration-limit 1)
set_tests_properties(cli_refinery_limited PROPERTIES WILL_FAIL TRUE)
add_test(NAME netlib_benchmarks COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/scripts/run_netlib.py --solver $<TARGET_FILE:markov-cero-solve> --instances afiro adlittle sc50a sc50b sc105 share2b recipe)
add_test(NAME miplib_benchmarks COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/scripts/run_miplib.py --solver $<TARGET_FILE:markov-cero-solve> --instances stein9 stein15 flugpl)
add_test(NAME gpu_benchmarks
  COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/scripts/run_gpu.py
    --solver $<TARGET_FILE:markov-cero-solve>
    --instances afiro blend sc50a sc50b
    --output ${CMAKE_BINARY_DIR}/reports/gpu_benchmark.csv)
add_test(NAME gpu_profiling
  COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/scripts/profile_gpu.py
    --solver $<TARGET_FILE:markov-cero-solve>
    --model ${CMAKE_SOURCE_DIR}/examples/blend.mps
    --output-json ${CMAKE_BINARY_DIR}/reports/profile_summary.json
    --output-md ${CMAKE_BINARY_DIR}/reports/profile_analysis.md)
set_tests_properties(
  build_info model_verifier mps_parser mps_fuzz_smoke model_properties
  dense_lu primal_simplex primal_simplex_properties dual_simplex
  warm_start_properties sparse_basis sparse_update_properties audit_regressions
  sparse_canonicalize presolve ruiz_scaling milp milp_heuristics milp_cuts
  strong_branching pdlp parallel_tree_search gpu_buffer equivalence gpu_reduction
  gpu_pdhg_step gpu_pdhg_restart gpu_pdhg_adaptive gpu_pdhg_kkt gpu_pdhg_timing
  qp json_records sovereignty_guard
  cli_blend_optimal cli_refinery_feasible cli_refinery_infeasible
  cli_refinery_malformed cli_refinery_limited
  netlib_benchmarks gpu_benchmarks gpu_profiling
  PROPERTIES
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "MARKOV_CERO_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
set_tests_properties(miplib_benchmarks
  PROPERTIES
    TIMEOUT 180
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "MARKOV_CERO_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
if(MARKOV_CERO_BUILD_FUZZER)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "libFuzzer target requires Clang")
  endif()
  add_executable(mps_fuzz tests/fuzz/mps_fuzz.cpp)
  target_link_libraries(mps_fuzz PRIVATE markov_cero_core)
  target_compile_options(mps_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(mps_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  add_executable(sparse_basis_fuzz tests/fuzz/sparse_basis_fuzz.cpp)
  target_link_libraries(sparse_basis_fuzz PRIVATE markov_cero_core)
  target_compile_options(sparse_basis_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(sparse_basis_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
