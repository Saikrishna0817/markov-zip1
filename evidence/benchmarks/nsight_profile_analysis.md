# markov-cero GPU Kernel Profiling & Roofline Analysis (SCALE_1000)

**Grounding:** Task T-5.14 / Decisive Design Principles `D-GPU-02` (Device Residency)
and `D-GPU-08` (Four-Part Timing).

## 1. Problem Dimensions & Telemetry

- **Rows / Constraints ($M$):** 525
- **Columns / Variables ($N$):** 980
- **Nonzero Matrix Coefficients ($NNZ$):** 2,884
- **PDHG Iterations Completed:** 160

## 2. Four-Part Timing Breakdown (`D-GPU-08`)

| Phase | Telemetry Key | Time (ms) | Fraction of Wall-Clock |
| :--- | :--- | :---: | :---: |
| Host-to-Device Transfer | `h2d_ms` | 0.040 ms | 2.2% |
| In-Device Compute Kernels | `kernel_ms` | 1.672 ms | 93.4% |
| Device-to-Host Download | `d2h_ms` | 0.001 ms | 0.1% |
| **End-to-End Wall Clock** | `total_ms` | **1.790 ms** | **100.0%** |

## 3. Proof of Device-Resident Loop (`D-GPU-02`)

- **Initial H2D Upload:** 156.59 KB (Matrix CSR, $A^T$ CSR, vectors, step sizes)
- **Final D2H Download:** 11.76 KB (primal and dual solution vectors)
- **In-Loop Host-Device Memory Transfers:** **0** (Strict Zero)
- **Residency Invariant:** Iterates remain device-resident across all iterations
  without host roundtrips.

## 4. Kernel Execution Share & Roofline Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **SpMV Share** ($A$ & $A^T$) | 69.6% | Streaming CSR matrix-vector products (warp-per-row) |
| **Primal Step & Proj.** | 17.7% | Elementwise axpy and box bound clamp |
| **Dual Step & Proj.** | 12.7% | Elementwise axpy and row dual projection |
| **Effective Compute** | 1.59 GFLOP/s | Sustained floating-point throughput |
| **Effective Bandwidth** | 18.05 GB/s | Memory streaming bandwidth utilization |
| **Arithmetic Intensity** | 0.0879 FLOP/B | Memory-bound regime (bandwidth-critical) |

## 5. Kernel Occupancy & Launch Configurations

- **SpMV (`spmv_csr_vector_kernel`):** 100% (256 threads/block, 24 regs/thread, 0 B smem)
- **Vector Ops (`pdhg_primal_step_kernel`):** 100% (256 threads/block, 22 regs/thread, 0 B smem)
- **Shared Memory per Block:** 0 bytes (no bank conflicts, max active blocks per SM)
- **Warp Synchronization:** Intra-warp shuffle reduction (`__shfl_down_sync`) with
  zero block-wide barriers in SpMV.
