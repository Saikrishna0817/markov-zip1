# GPU Acceleration Strategy & Phase 5 Roadmap

> **The 90-Second Executive Summary:**  
> In `markov-cero` v0.5.2, **GPU acceleration is not implemented**. The solver core is strictly  
> sovereign CPU C++20. However, the system was intentionally architected with a GPU on-ramp:  
> the CPU matrix-free PDLP engine (`src/lp/first_order/pdlp.cpp`). Simplex is the mathematically  
> wrong target for GPU parallelization; first-order primal-dual methods (PDHG/PDLP) are the  
> correct target. Phase 5 delivers this acceleration cleanly and honestly.

---

## 1. Current Status (v0.5.2)

- **Solver Core**: 100% CPU sovereign C++20 with standard threading (`std::jthread`).
- **External Dependencies**: Strict `{C++20 stdlib, Threads}`. Zero CUDA or third-party links.
- **Telemetry Disclosure**: All JSON outputs emit:
  `"limitations":"CPU sovereign LP and MILP Branch-and-Cut engine; GPU and QP are not implemented."`

---

## 2. Why Simplex is the Wrong GPU Target

The problem statement asks for "GPU-Accelerated" optimization. In linear programming, attempting
to port the Revised Simplex algorithm directly to GPUs is a well-known anti-pattern in the
optimization literature:

1. **Sequential Dependencies**: Simplex is inherently serial. Each basis change requires
   pivoting, basis update, and pricing. One pivot depends entirely on the previous basis.
2. **Irregular Sparse Triangular Solves**: Computing $B^{-1} a_j$ (FTRAN) and $B^{-T} c_B$ (BTRAN)
   involves irregular pointer-chasing through sparse LU factorizations and Eta matrices. These
   suffer from severe thread divergence and poor memory coalescence on SIMT architectures.
3. **Data Transfer Bottleneck**: Re-uploading basis matrices or transferring pivots between
   host and device dominates runtime, erasing any compute advantage.
4. **Literature Consensus**: Research (e.g., Hall 2010, Bixby 2012) demonstrates that modern
   CPU cache hierarchies and SIMD vectorization outperform GPU simplex implementations on
   standard benchmarks.

---

## 3. Why PDLP (First-Order Primal-Dual) is the Right GPU Target

Primal-Dual Hybrid Gradient (PDHG / PDLP; Applegate et al., NeurIPS 2021; Lu & Yang, cuPDLP.jl,
2023) fundamentally alters GPU feasibility:

1. **Regular, High-Throughput Compute**: Every PDHG iteration consists strictly of:
   - Sparse matrix-vector product (SpMV): $q = A x$
   - Transpose sparse matrix-vector product: $s = A^T y$
   - Elementwise vector axpy, scaling, and bound projections
   - Deterministic parallel reductions (inner products, 2-norms, $\infty$-norms)
2. **Zero In-Loop Memory Transfers**: The matrix $A$ is uploaded once to device memory in CSR
   and CSR-of-$A^T$ formats. All iterate vectors ($x, y, \bar{x}, \bar{y}$) remain device-resident.
   The host receives only scalar convergence telemetry per check interval.
3. **Designed CPU On-Ramp**: `markov-cero` already implements matrix-free PDLP on CPU
   (`src/lp/first_order/pdlp.cpp`). Phase 5 does not invent a new algorithm; it ports this
   validated iteration kernel to CUDA.

---

## 4. Phase 5 Architecture & Decisions

Phase 5 introduces optional GPU acceleration with zero disruption to the non-GPU build:

- **`D-GPU-01` (Target)**: Restarted PDHG, not simplex or branch-and-bound. Avoids serial
  bottlenecks and leverages streaming SpMV.
- **`D-GPU-02` (Residency)**: All iterates stay device-resident. Zero H2D/D2H memory transfers
  inside the iteration loop.
- **`D-GPU-03` (Sovereign SpMV)**: Custom warp-per-row CSR SpMV kernel. cuSPARSE is used only
  as an external benchmark reference, never linked into the shipped solver.
- **`D-GPU-04` (Adaptive Restarts)**: Restarts on normalized duality gap drop iteration count
  by orders of magnitude (Applegate et al. 2023).
- **`D-GPU-05` (Stepsize & Weight)**: Malitsky–Pock adaptive stepsize with primal/dual weight
  updates for robust convergence.
- **`D-GPU-06` (Preconditioning)**: Reuses CPU Ruiz equilibration (`src/scale/ruiz_scaling.cpp`)
  and Chambolle–Pock diagonal preconditioning.
- **`D-GPU-07` (Precision)**: FP64 by default for defensible feasibility tolerances ($\le 10^{-6}$).
  FP32 is permitted only as a labelled ablation with FP64 residual verification.
- **`D-GPU-08` (Timing Accounting)**: Mandatory four-part timing emitted in JSON telemetry:
  `h2d_ms`, `kernel_ms`, `d2h_ms`, and `total_ms`. Never report kernel time alone.
- **`D-GPU-09` (Termination)**: Relative KKT error evaluation across primal residual, dual residual,
  and duality gap at separate tolerances ($10^{-4}, 10^{-6}, 10^{-8}$).
- **`D-GPU-10` (Build Flag)**: Optional CMake flag `-DMARKOV_CERO_ENABLE_CUDA=ON`. Without it,
  the project builds and passes all tests without a GPU.
- **`D-GPU-11` (Equivalence)**: CPU/GPU equivalence test is a hard merge gate.
- **`D-GPU-12` (Minimal Interface)**: Solver integration is a single option:
  `struct Options { Backend backend = Backend::cpu; };` in `pdlp.hpp`.

### Module Structure

```text
gpu/
├── include/markov_cero/gpu/
│   ├── device.hpp          # Device query, capability check, graceful fallback
│   ├── buffer.hpp          # DeviceBuffer<T>: RAII, zero-copy, explicit transfers
│   ├── csr.hpp             # DeviceCsr: values, col_idx, row_ptr from SparseCsc
│   └── kernels.hpp         # SpMV, vector axpy, bound projections, reductions
├── src/
│   ├── device.cpp  buffer.cpp  csr.cpp
├── kernels/
│   ├── spmv.cu  vector_ops.cu  reduce.cu  pdhg_step.cu
└── tests/
    ├── equivalence_test.cpp # Kernel-level validation against CPU implementation
    └── pdhg_gpu_test.cpp    # Full solve equivalence and tolerance verification
```

---

## 5. Headline Result & Honest Negative Results

A credible optimization submission reports where GPU acceleration wins **and where it loses**:

1. **The Crossover Study**:
   - Small LPs ($< 10^4$ nonzeros, e.g., Netlib `AFIRO` or `SC50A`): CPU Simplex solves in $< 2$ ms.
     GPU kernel launch latency and PCIe transfer overhead make GPU PDLP slower.
   - Large LPs ($10^5$ to $10^7$ nonzeros): SIMT streaming bandwidth dominates. GPU PDLP achieves
     multi-fold speedups over single-threaded CPU simplex and CPU PDLP.
2. **Documented Negative Results**:
   - Highly degenerate LPs with massive constraint counts and low nonzero density favor simplex.
   - Ill-conditioned constraint systems require high iteration counts where simplex basis
     factorization remains more accurate.
   - All negative findings will be documented with exact timings in Phase 5 benchmarks.
