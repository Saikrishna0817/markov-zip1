# SIHOpt — Research Papers, Mathematical Foundations, and Design Lineage

> **Smart India Hackathon 2026 — PS SIH26119 (MRPL)**  
> *Indigenous GPU-Accelerated Software Optimization Solver (Sovereign Alternative to Xpress / CPLEX)*  
> Document Version: 1.0.0 · Classification: Technical Reference & Literature Registry

---

## Table of Contents

1. [Executive Summary & Clean-Room Lineage](#1-executive-summary--clean-room-lineage)
2. [Deep Dive: Parallel Dual Revised Simplex (Huangfu & Hall, 2018)](#2-deep-dive-parallel-dual-revised-simplex-huangfu--hall-2018)
   - 2.1 Standard Form & Dual Formulation
   - 2.2 The Three Iteration Components: CHUZR, CHUZC, Update
   - 2.3 Dual Steepest-Edge (DSE) & Forrest–Goldfarb Recurrence
   - 2.4 Harris Two-Pass Ratio Test & Bound-Flipping Ratio Test (BFRT)
   - 2.5 Basis Updates: Product-Form Eta vs. Forrest–Tomlin
   - 2.6 Hyper-Sparsity & Gilbert–Peierls Graph Reach
   - 2.7 PAMI (Parallel Alternating Method for Iterations)
   - 2.8 SIP (Single-Iteration Parallelism)
3. [Presolve & Reversible Matrix Transformations](#3-presolve--reversible-matrix-transformations)
   - 3.1 Mathematical Foundations (Andersen & Andersen 1995; Achterberg et al. 2020)
   - 3.2 The Eight Core Reductions
   - 3.3 The Invertible Postsolve Stack
4. [First-Order Methods & GPU Acceleration (PDHG / PDLP)](#4-first-order-methods--gpu-acceleration-pdhg--pdlp)
   - 4.1 Practical Large-Scale LP via PDHG (Applegate et al. 2021)
   - 4.2 Saddle-Point Formulation & Chambolle–Pock Iterations
   - 4.3 Diagonal Preconditioning (Ruiz + Pock–Chambolle)
   - 4.4 Adaptive Restarts & Primal-Dual Infeasibility Rays
   - 4.5 GPU Architecture Mapping (cuPDLP-C / cuPDLP+)
5. [Mixed-Integer Linear Programming (MILP) & Branch-and-Bound](#5-mixed-integer-linear-programming-milp--branch-and-bound)
   - 5.1 Branch-and-Bound Architecture & Dual Simplex Warm-Starts
   - 5.2 Branching Variable Selection: Reliability Branching (Achterberg et al. 2005)
   - 5.3 Cutting Planes: Gomory Fractional Cuts from Tableau
   - 5.4 Node Selection & Primal Diving Heuristics
6. [Convex Quadratic Programming (QP) & ADMM](#6-convex-quadratic-programming-qp--admm)
   - 6.1 Formulation & KKT Conditions
   - 6.2 OSQP Operator Splitting (Stellato et al. 2020)
   - 6.3 Quasi-Definite Systems & Infeasibility Certificates
7. [Independent Verification & Exact Certificates](#7-independent-verification--exact-certificates)
   - 7.1 Farkas’ Lemma & Infeasibility Proofs
   - 7.2 Independent Verification Architecture (Applegate et al. 2007; Cheung et al. 2017)
   - 7.3 Scaled Residual Metrics & Tolerances
8. [Refinery Domain Optimization (MRPL Operational Models)](#8-refinery-domain-optimization-mrpl-operational-models)
   - 8.1 Crude Distillation Unit (CDU) Cut-Yield Formulations
   - 8.2 Product Blending, Sulfur Give-Away, and RVP Constraints
   - 8.3 Shadow Prices (Dual Values) & Refinery Economics
9. [Numerical Integrity, IEEE 754 & Secure Engineering](#9-numerical-integrity-ieee-754--secure-engineering)
   - 9.1 Floating-Point Arithmetic & Error Propagation (Higham 2002)
   - 9.2 Fail-Closed Numerical Contracts
   - 9.3 Hardened Software Engineering (CWE Mitigations)
10. [Comprehensive Annotated Bibliography](#10-comprehensive-annotated-bibliography)

---

## 1. Executive Summary & Clean-Room Lineage

The SIHOpt project is an indigenous, sovereign mathematical optimization engine developed from first principles in C++20 for Smart India Hackathon 2026 (Problem Statement **SIH26119**, sponsored by **Mangalore Refinery and Petrochemicals Limited - MRPL**).

### Clean-Room Engineering Principle
SIHOpt does **not** wrap, copy, fork, or translate source code, test suites, internal naming, or constants from existing solvers (such as HiGHS, GLPK, Clp, SCIP, CPLEX, Gurobi, or Xpress). Instead:
- All algorithms are implemented directly from peer-reviewed mathematical literature and textbooks.
- Software boundaries are enforced mechanically in CI via `scripts/no_solver_guard.py` and link-line inspection.
- Correctness is proven by an **independent verifier** that shares zero linear algebra or state with the solver core, checking mathematical certificates in original model space.

---

## 2. Deep Dive: Parallel Dual Revised Simplex (Huangfu & Hall, 2018)

> **Primary Source**:  
> Qi Huangfu and J. A. Julian Hall (2018). *"Parallelizing the dual revised simplex method"*, **Mathematical Programming Computation**, 10(1): 119–142.  
> DOI: [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5)

Huangfu & Hall (2018) is the foundational research paper behind modern high-performance dual revised simplex codes (including the engine implemented in FICO Xpress and later open-sourced as HiGHS). It addresses the classical bottleneck: while the revised simplex method has sequential data dependencies, parallel speedup can be achieved via **PAMI** (Parallel Alternating Method for Iterations) and **SIP** (Single-Iteration Parallelism).

### 2.1 Standard Form & Dual Formulation

Consider the linear program in bounded standard form:
$$\begin{aligned}
\min \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & A \mathbf{x} = \mathbf{b}, \\
& \mathbf{l} \le \mathbf{x} \le \mathbf{u},
\end{aligned}$$
where $A \in \mathbb{R}^{m \times n}$ has full row rank $m$.

A basis partition splits columns into basic indices $\mathcal{B}$ ($|\mathcal{B}| = m$) and nonbasic indices $\mathcal{N}$ ($|\mathcal{N}| = n - m$):
$$A = [B \quad N], \quad \mathbf{x} = \begin{bmatrix} \mathbf{x}_B \\ \mathbf{x}_N \end{bmatrix}, \quad \mathbf{c} = \begin{bmatrix} \mathbf{c}_B \\ \mathbf{c}_N \end{bmatrix}.$$

The basic variables and reduced costs are given by:
$$\mathbf{x}_B = B^{-1} (\mathbf{b} - N \mathbf{x}_N),$$
$$\widehat{\mathbf{c}}_N^T = \mathbf{c}_N^T - \mathbf{c}_B^T B^{-1} N = \mathbf{c}_N^T - \boldsymbol{\pi}^T N,$$
where $\boldsymbol{\pi}^T = \mathbf{c}_B^T B^{-1}$ is the vector of dual simplex multipliers (computed via **BTRAN**: $B^T \boldsymbol{\pi} = \mathbf{c}_B$).

**Dual Feasibility Condition**:
For nonbasic variables $j \in \mathcal{N}$:
$$\begin{cases}
\widehat{c}_j \ge 0 & \text{if } x_j = l_j \text{ (at lower bound)}, \\
\widehat{c}_j \le 0 & \text{if } x_j = u_j \text{ (at upper bound)}, \\
\widehat{c}_j = 0 & \text{if } l_j < x_j < u_j \text{ (free / boxed variable inside)}.
\end{cases}$$

The dual simplex method maintains dual feasibility at all times ($\widehat{\mathbf{c}}_N$ satisfies the above signs) while moving towards primal feasibility ($\mathbf{l}_B \le \mathbf{x}_B \le \mathbf{u}_B$).

---

### 2.2 The Three Iteration Components: CHUZR, CHUZC, Update

Each iteration executes three phases:

```
[1. Optimality Test: CHUZR]
      │ Choose leaving variable p ∈ B with greatest weighted infeasibility
      ▼
   [BTRAN]
      │ Solve Bᵀ ê_p = e_p
      ▼
   [SPMV]
      │ Compute tableau row â_pᵀ = ê_pᵀ A
      ▼
[2. Ratio Test: CHUZC]
      │ Choose entering variable q ∈ N via Harris / BFRT
      ▼
   [FTRAN]
      │ Solve B â_q = a_q
      ▼
 [FTRAN-DSE]
      │ Solve B v = ê_p
      ▼
[3. Update]
      │ Swap p ↔ q, update x_B, ĉ_N, B⁻¹, and DSE weights
      ▼
(Next iteration)
```

1. **CHUZR (Choice of Row / Leaving Variable)**:
   Find basic variable $p \in \mathcal{B}$ violating its bounds $x_p < l_p$ or $x_p > u_p$.
   Under dual steepest-edge pricing:
   $$p = \arg\max_{i \in \mathcal{B}} \frac{(\Delta x_i)^2}{\gamma_i},$$
   where $\Delta x_i = \max(l_i - x_i, x_i - u_i, 0)$ is the primal infeasibility and $\gamma_i$ is the dual steepest-edge weight.

2. **CHUZC (Choice of Column / Entering Variable)**:
   Compute the row of the reduced tableau $\widehat{\mathbf{a}}_p^T$:
   - **BTRAN**: Solve $B^T \widehat{\mathbf{e}}_p = \mathbf{e}_p$.
   - **SPMV**: Form $\widehat{\mathbf{a}}_p^T = \widehat{\mathbf{e}}_p^T A$.
   - **Ratio Test**: Determine $q \in \mathcal{N}$ minimizing the step size $\theta = \frac{|\widehat{c}_j|}{|\widehat{a}_{pj}|}$ for eligible nonbasic directions that maintain dual feasibility.

3. **Update**:
   - **FTRAN**: Solve $B \widehat{\mathbf{a}}_q = \mathbf{a}_q$.
   - Update primal values: $\mathbf{x}_B \leftarrow \mathbf{x}_B - \theta \widehat{\mathbf{a}}_q$.
   - Update dual reduced costs: $\widehat{\mathbf{c}}_N \leftarrow \widehat{\mathbf{c}}_N - \alpha \widehat{\mathbf{a}}_p$.
   - Update basis representation $B^{-1}$ with column replacement.
   - Update DSE weights $\gamma_i$.

---

### 2.3 Dual Steepest-Edge (DSE) & Forrest–Goldfarb Recurrence

> **Key Finding from Audit**:  
> In SIHOpt v0.5.1, steepest-edge pricing recomputed the full vector norm $\|B^{-T} \mathbf{e}_i\|_2$ by performing BTRAN and SpMV for each candidate row in $O(m^2)$ work per pivot!  
> **Huangfu & Hall (2018) Section 2.2.1** and **Forrest & Goldfarb (1992)** establish the exact $O(m)$ recurrence update that eliminates this bottleneck.

#### Mathematical Definition
The dual steepest-edge weight for row $i \in \mathcal{B}$ is the Euclidean norm of row $i$ of the basis inverse:
$$\gamma_i = \|\widehat{\mathbf{e}}_i^T\|_2^2 = \|B^{-T} \mathbf{e}_i\|_2^2.$$

#### The Forrest–Goldfarb Update Formula
When variable $p \in \mathcal{B}$ leaves and $q \in \mathcal{N}$ enters:
1. Compute vector $\mathbf{v} \in \mathbb{R}^m$ via one additional **FTRAN-DSE** solve:
   $$B \mathbf{v} = \widehat{\mathbf{e}}_p \quad \iff \quad \mathbf{v} = B^{-1} B^{-T} \mathbf{e}_p.$$
2. The pivot element is $\alpha = \widehat{a}_{pq}$.
3. For all basic rows $i \in \mathcal{B} \setminus \{p\}$:
   $$\gamma_i \leftarrow \gamma_i - 2 \left(\frac{\widehat{a}_{iq}}{\alpha}\right) v_i + \left(\frac{\widehat{a}_{iq}}{\alpha}\right)^2 \gamma_p.$$
4. For the pivot row $p$:
   $$\gamma_p \leftarrow \frac{\gamma_p}{\alpha^2}.$$
5. Numerical Safeguard: Because cancellation can cause computed $\gamma_i$ to drift or become negative, if $\gamma_i < \epsilon_{\text{DSE}}$, reinitialize $\gamma_i = 1.0$, or recompute exact weights every $K$ pivots (typically $K = 500$).

This replaces an $O(m^2)$ search with a single FTRAN solve ($O(\text{flops})$) and an $O(m)$ vector update!

---

### 2.4 Harris Two-Pass Ratio Test & Bound-Flipping Ratio Test (BFRT)

#### Paula Harris Two-Pass Ratio Test (1973)
To prevent numerical instability caused by near-zero pivots $\widehat{a}_{pj}$:
1. **Pass 1**: Compute the maximum allowable dual step $\theta_{\max}$ using a relaxed tolerance $\delta > 0$:
   $$\theta_{\max} = \min_{j \in \mathcal{N}} \left\{ \frac{\widehat{c}_j + \delta}{|\widehat{a}_{pj}|} : \text{sign eligible} \right\}.$$
2. **Pass 2**: Among all nonbasics with step $\theta_j \le \theta_{\max}$, choose the entering variable $q$ that **maximizes the pivot magnitude**:
   $$q = \arg\max_{j \in \mathcal{N}, \theta_j \le \theta_{\max}} |\widehat{a}_{pj}|.$$
This achieves numerical stability while strictly preventing small pivots.

#### Bound-Flipping Ratio Test (BFRT)
For box-constrained problems ($l_j \le x_j \le u_j$), multiple nonbasic variables can change from their lower bound to their upper bound (or vice-versa) during a single pivot without entering the basis, absorbing infeasibility:
- Reduces iteration counts by **2x to 5x** on bounded industrial LPs.
- The step continues flipping nonbasics until the primal infeasibility of row $p$ is zeroed or minimized.

---

### 2.5 Basis Updates: Product-Form Eta vs. Forrest–Tomlin

A simplex pivot replaces the $p$-th column of $B$ with $\mathbf{a}_q$:
$$\bar{B} = B + (\mathbf{a}_q - B \mathbf{e}_p) \mathbf{e}_p^T = B (I + (\widehat{\mathbf{a}}_q - \mathbf{e}_p) \mathbf{e}_p^T).$$

#### Product-Form Eta Matrix
The inverse is updated by appending an elementary matrix $E$:
$$\bar{B}^{-1} = E B^{-1}, \quad E = I - \frac{1}{\widehat{a}_{pq}} (\widehat{\mathbf{a}}_q - \mathbf{e}_p) \mathbf{e}_p^T.$$
- **FTRAN** applies etas chronologically: $E_k \cdots E_2 E_1 B_0^{-1} \mathbf{b}$.
- **BTRAN** applies etas in reverse transpose order: $B_0^{-T} E_1^T E_2^T \cdots E_k^T \mathbf{c}$.
- When eta fill accumulates ($k > 64$ or density $> 30\%$), $B$ is refactorized via fresh sparse LU.

#### Forrest–Tomlin (FT) Update
Instead of an arbitrary eta column, FT permutes row $p$ to the bottom of triangular factor $U$, creating an upper Hessenberg matrix, and eliminates the row subdiagonal entries with row operations. This keeps $U$ triangular and strictly bounds fill-in.

---

### 2.6 Hyper-Sparsity & Gilbert–Peierls Graph Reach

> **Reference**:  
> J. A. J. Hall and K. I. M. McKinnon (2005). *"Hyper-sparsity in the revised simplex method and how to exploit it"*, **Comput. Optim. Appl.**, 32(3): 259–283.

In large-scale sparse LPs, vectors $\mathbf{e}_p$ and $\mathbf{a}_q$ often have sparsity $< 0.1\%$. If $B^{-1}$ is also sparse, the solution vector $\mathbf{x}$ will have very few nonzeros ($k \ll m$).
- Standard triangular solve takes $O(m)$ by scanning all rows.
- **Gilbert–Peierls Algorithm**: The nonzeros in the solution $\mathbf{x}$ to $L \mathbf{x} = \mathbf{b}$ correspond precisely to the nodes in the directed acyclic graph (DAG) of $L$ reachable from the nonzero indices of $\mathbf{b}$.
- Traverses only reachable nodes via depth-first search (DFS), topologically sorts them, and executes arithmetic in $O(\text{flops})$ without touching the rest of the $m$-dimensional vector.

---

### 2.7 PAMI (Parallel Alternating Method for Iterations)

Huangfu & Hall (2018) Section 3 introduces **PAMI** to parallelize across iterations using dual suboptimization:

```
PAMI Algorithm Structure:
=========================
1. Major Optimality Test:
   Select s candidates P = {p₁, p₂, ..., pₛ} ⊆ B using DSE (where s = thread count).
2. Minor Initialization (Task Parallel):
   Compute BTRAN in parallel across s threads:
   Thread k solves: Bᵀ ê_{p_k} = e_{p_k}.
3. Minor Iterations:
   Execute minor simplex pivots using candidate rows:
   - SPMV: â_{p_k}ᵀ = ê_{p_k}ᵀ A (parallelized across column partitions)
   - Ratio test (CHUZC1): parallel reduction across partitions
   - Select entering variable q_k
   - Update minor tableau
4. Major Update:
   Apply accumulated pivots back to global basis representation.
```

---

### 2.8 SIP (Single-Iteration Parallelism)

For instances where PAMI causes iteration growth, Huangfu & Hall (2018) Section 4 defines **SIP**:
- **Thread 1**: Computes FTRAN of $\mathbf{a}_q$ ($B \widehat{\mathbf{a}}_q = \mathbf{a}_q$).
- **Thread 2**: Computes FTRAN-DSE of $\widehat{\mathbf{e}}_p$ ($B \mathbf{v} = \widehat{\mathbf{e}}_p$).
- **Thread 3 & 4**: Computes FTRAN-BFRT for bound flipping and parallel SpMV.
- Components execute concurrently with synchronization points at ratio test and basis update.

---

## 3. Presolve & Reversible Matrix Transformations

> **Primary Sources**:  
> 1. Erling D. Andersen and Knud D. Andersen (1995). *"Presolving in linear programming"*, **Mathematical Programming**, 71(2): 221–245.  
> 2. Tobias Achterberg et al. (2020). *"Presolve reductions in mixed integer programming"*, **INFORMS J. Comput.**, 32(2): 473–506.

Presolve reduces problem dimensions and tightens bounds before sending the model to the simplex or interior-point engine. Industrial models typically shrink by **30% to 70%** in nonzero count.

### 3.1 The Eight Core Reductions

| Reduction | Condition | Mathematical Transformation | Inverse (Postsolve) Action |
|---|---|---|---|
| **1. Empty Rows** | $a_{ij} = 0 \; \forall j$ | If $0 \in [b_i^{\min}, b_i^{\max}]$, remove row; else declare **Infeasible**. | None (discarded). |
| **2. Empty Columns** | $a_{ij} = 0 \; \forall i$ | If $c_j > 0$, fix $x_j = l_j$; if $c_j < 0$, fix $x_j = u_j$; if $c_j = 0$, fix $x_j = 0$. | Set $x_j$ to fixed bound; dual $\widehat{c}_j = c_j$. |
| **3. Fixed Variables** | $l_j = u_j = \bar{x}_j$ | Remove variable $x_j$; shift RHS $\mathbf{b} \leftarrow \mathbf{b} - A_{\cdot j} \bar{x}_j$; objective offset $z_0 \leftarrow z_0 + c_j \bar{x}_j$. | Restore $x_j = \bar{x}_j$; compute $\widehat{c}_j = c_j - \boldsymbol{\pi}^T A_{\cdot j}$. |
| **4. Row Singletons** | Row $i$ has one nonzero $a_{ik}$ | Inequality $l_i \le a_{ik} x_k \le u_i$ directly implies new variable bounds: $\frac{l_i}{a_{ik}} \le x_k \le \frac{u_i}{a_{ik}}$. Intersect with $[l_k, u_k]$. If empty, **Infeasible**. Remove row $i$. | Restore row $i$; compute dual multiplier $\pi_i = \frac{c_k - \sum_{r \ne i} \pi_r a_{rk}}{a_{ik}}$. |
| **5. Column Singletons** | Col $k$ has one nonzero $a_{ik}$ (slack-like) | Express $x_k = \frac{b_i - \sum_{j \ne k} a_{ij} x_j}{a_{ik}}$. Substitute $x_k$ into objective and bound inequalities. | Recompute $x_k$ from optimal nonbasic and basic values of row $i$. |
| **6. Free Variable Substitution** | Variable $x_k$ has $-\infty < x_k < \infty$ | Use equality constraint $i$ where $a_{ik} \ne 0$ to eliminate $x_k = \frac{b_i - \sum_{j \ne k} a_{ij} x_j}{a_{ik}}$. Eliminate row $i$ and col $k$. | Restore $x_k$ from row $i$; dual $\pi_i = c_k / a_{ik}$. |
| **7. Implied Bound Tightening** | Row $i$: $\sum a_{ij} x_j \le b_i$ | Activity bounds: $L_i = \sum_{j: a_{ij}>0} a_{ij} l_j + \sum_{j: a_{ij}<0} a_{ij} u_j$. For each $k$: $x_k \le \frac{b_i - L_i + a_{ik} l_k}{a_{ik}}$. | Update original bounds. |
| **8. Duplicate Rows** | Row $r = \alpha \cdot \text{Row } s$ | If RHS is consistent, merge bounds into single row; if inconsistent, **Infeasible**. | Split dual multiplier between original rows: $\pi_r + \alpha \pi_s = \pi_{\text{merged}}$. |

### 3.2 The Invertible Postsolve Stack

```
[Original Model (A, b, c, l, u)]
      │
      ▼
[Presolve Engine] ──(Push Reduction Record)──► [LIFO Postsolve Stack]
      │
      ▼
[Reduced Model (A', b', c', l', u')]
      │
      ▼
[Solver Core: Simplex / IPM / PDLP]
      │
      ▼
[Reduced Solution (x', π', ĉ')]
      │
      ▼
[Postsolve Reconstructor] ◄──(Pop Records in Reverse)─── [LIFO Stack]
      │
      ▼
[Full Original Solution (x, π, ĉ)]
      │
      ▼
[Independent Verifier (Checks against Original Model)]
```

**Cardinal Rule**: The postsolved solution MUST be re-measured against the original unscaled model. Any discrepancy surfaces as an objective/feasibility violation.

---

## 4. First-Order Methods & GPU Acceleration (PDHG / PDLP)

> **Primary Sources**:  
> 1. David Applegate, Mateo Díaz, Oliver Hinder, Haihao Lu, Miles Lubin, Brendan O'Donoghue, Warren Schudy (Google Research, 2021). *"Practical Large-Scale Linear Programming Using Primal-Dual Hybrid Gradient"*, arXiv:2106.04756.  
> 2. W. Dong et al. (2023). *"cuPDLP: A GPU Implementation of the Primal-Dual Hybrid Gradient Algorithm for Linear Programming"*.

For massive LPs ($10^6$ to $10^8$ variables) that exceed CPU memory or sparse LU factorization capacity, first-order methods running on GPUs represent the current state-of-the-art.

### 4.1 Saddle-Point Formulation

The standard LP:
$$\min_{\mathbf{x} \in \mathcal{K}} \mathbf{c}^T \mathbf{x} \quad \text{s.t.} \quad A \mathbf{x} = \mathbf{b},$$
is reformulated into a minimax saddle-point problem:
$$\min_{\mathbf{x} \ge \mathbf{0}} \max_{\mathbf{y} \in \mathbb{R}^m} \mathcal{L}(\mathbf{x}, \mathbf{y}) \equiv \mathbf{c}^T \mathbf{x} + \mathbf{y}^T (\mathbf{b} - A \mathbf{x}).$$

### 4.2 Chambolle–Pock (PDHG) Iterations

With primal step size $\tau > 0$ and dual step size $\sigma > 0$ satisfying $\tau \sigma \|A\|_2^2 < 1$:
$$\begin{aligned}
\mathbf{x}^{k+1} &= \text{proj}_{[\mathbf{l}, \mathbf{u}]} \left( \mathbf{x}^k - \tau (\mathbf{c} - A^T \mathbf{y}^k) \right), \\
\bar{\mathbf{x}}^{k+1} &= 2 \mathbf{x}^{k+1} - \mathbf{x}^k \quad (\text{extrapolation step}), \\
\mathbf{y}^{k+1} &= \mathbf{y}^k + \sigma (\mathbf{b} - A \bar{\mathbf{x}}^{k+1}).
\end{aligned}$$

Every operation in this loop is either:
1. Matrix-vector multiplication ($A \bar{\mathbf{x}}$ and $A^T \mathbf{y}$), or
2. Element-wise vector addition, scaling, and clamping ($\text{proj}$).

**No linear system factorization is required.** This algorithm maps naturally to parallel GPU hardware.

### 4.3 Diagonal Preconditioning (Ruiz + Pock–Chambolle)

Raw industrial matrices have widely disparate row and column norms.
1. **Ruiz Equilibration**: Compute diagonal matrices $D_R \in \mathbb{R}^{m \times m}$ and $D_C \in \mathbb{R}^{n \times n}$ such that every row and column of $\tilde{A} = D_R A D_C$ has unit $\ell_\infty$ norm.
2. **Pock–Chambolle Step Sizing**:
   $$\tau_j = \frac{1}{\sum_{i=1}^m |\tilde{a}_{ij}|}, \quad \sigma_i = \frac{1}{\sum_{j=1}^n |\tilde{a}_{ij}|}.$$
   This guarantees $\tau_j \sigma_i \|\tilde{a}_{ij}\|^2 \le 1$, ensuring theoretical convergence without expensive power-iteration eigenvalue solves.

### 4.4 Adaptive Restarts & Termination

- **Normalized Duality Gap**:
  $$\text{Gap}(\mathbf{x}, \mathbf{y}) = \frac{|\mathbf{c}^T \mathbf{x} - \mathbf{b}^T \mathbf{y}|}{1 + |\mathbf{c}^T \mathbf{x}| + |\mathbf{b}^T \mathbf{y}|}.$$
- **Primal Residual**: $\|\mathbf{b} - A \mathbf{x}\|_2 / (1 + \|\mathbf{b}\|_2) \le \epsilon_{\text{tol}}$.
- **Dual Residual**: $\|\mathbf{c} - A^T \mathbf{y} - \mathbf{s}\|_2 / (1 + \|\mathbf{c}\|_2) \le \epsilon_{\text{tol}}$.
- **Restart Strategy**: Restart the sequence from the ergodic (running average) iterate $\bar{\mathbf{x}} = \frac{1}{K} \sum_{k=1}^K \mathbf{x}^k$ whenever the normalized gap decreases by a factor $\beta \in (0, 1)$ or after a fixed iteration limit.

### 4.5 GPU Architecture Mapping (cuPDLP Style)

```
GPU Memory Hierarchy & Kernel Pipeline:
┌────────────────────────────────────────────────────────────────┐
│ Device Global Memory:                                          │
│   - Matrix A in CSR format (row pointers, col indices, values) │
│   - Matrix Aᵀ in CSC format (col pointers, row indices, values)│
│   - Primal vectors x, x_bar, l, u, c                           │
│   - Dual vectors y, b                                          │
├────────────────────────────────────────────────────────────────┤
│ Streaming Multiprocessors (SMs):                               │
│   Kernel 1: cuSPMV_A_transpose  --> computes Aᵀ y             │
│   Kernel 2: Elementwise_PrimalUpdate (Fused axpy + clamp)      │
│   Kernel 3: cuSPMV_A            --> computes A x_bar           │
│   Kernel 4: Elementwise_DualUpdate (Fused axpy)                │
│   Kernel 5: DotProduct_NormReduction (CUB block-reduction)     │
└────────────────────────────────────────────────────────────────┘
Zero CPU-GPU synchronization inside inner iterations.
Convergence residuals checked only every 64 or 128 iterations.
```

---

## 5. Mixed-Integer Linear Programming (MILP) & Branch-and-Bound

> **Primary Sources**:  
> 1. Tobias Achterberg, Thorsten Koch, Alexander Martin (2005). *"Branching rules revisited"*, **Operations Research Letters**, 33(1): 42–54.  
> 2. Robert E. Bixby et al. (2000). *"MIP: Theory and Practice — Closing the Gap"*.

A Mixed-Integer Linear Program (MILP) minimizes $\mathbf{c}^T \mathbf{x}$ subject to linear constraints where a subset of variables $x_j \in \mathbb{Z}$ ($j \in \mathcal{I}$).

### 5.1 Branch-and-Bound Tree & Dual Simplex Warm-Starts

The branch-and-bound algorithm searches a tree of continuous LP relaxations:
1. Solve root LP relaxation ($x_j \in \mathbb{R}$ for all $j$).
2. If all $x_j \in \mathbb{Z}$ ($j \in \mathcal{I}$), root is integer optimal.
3. Otherwise, select fractional candidate $x_k = f \notin \mathbb{Z}$ ($k \in \mathcal{I}$).
4. Create two children:
   - Left branch: Add bound $x_k \le \lfloor f \rfloor$.
   - Right branch: Add bound $x_k \ge \lceil f \rceil$.
5. **Warm-Start Advantage**: Adding a bound constraint leaves the parent basis **dual feasible**, but primal infeasible. The **dual simplex method** re-optimizes the child node in **5 to 50 pivots**, whereas solving from scratch would take thousands!

### 5.2 Reliability Branching (Achterberg et al., 2005)

Variable selection is the single most critical factor in search-tree size:
- **Strong Branching**: Tentatively execute dual simplex iterations on both child branches for all fractional candidates to measure objective change $\Delta z_j^-$ and $\Delta z_j^+$. Very accurate, but computationally expensive.
- **Pseudo-Cost Branching**: Track historical average objective improvement per unit step:
  $$\Psi_j^- = \frac{\sum \Delta z_j^-}{\sum f_j - \lfloor f_j \rfloor}, \quad \Psi_j^+ = \frac{\sum \Delta z_j^+}{\sum \lceil f_j \rceil - f_j}.$$
  Very fast, but unreliable early in the search.
- **Reliability Branching (The Hybrid Standard)**:
  - If variable $j$ has been branched on fewer than $\eta_{\text{rel}}$ times (typically $\eta_{\text{rel}} = 8$), execute **strong branching** and update pseudo-costs.
  - Once variable $j$ reaches $\eta_{\text{rel}}$ observations, trust its pseudo-costs.
  - Score candidates via product score:
    $$\text{Score}_j = (1 - \mu) \min(\Delta z_j^-, \Delta z_j^+) + \mu \max(\Delta z_j^-, \Delta z_j^+) + \epsilon, \quad \mu \approx 0.16.$$

### 5.3 Gomory Fractional Cutting Planes

Given an optimal simplex tableau row for basic integer variable $x_i$:
$$x_i + \sum_{j \in \mathcal{N}} \bar{a}_{ij} x_j = \bar{b}_i, \quad \bar{b}_i \notin \mathbb{Z}.$$
Decompose each coefficient into integer and fractional parts: $\bar{a}_{ij} = \lfloor \bar{a}_{ij} \rfloor + f_{ij}$, $\bar{b}_i = \lfloor \bar{b}_i \rfloor + f_0$ where $f_{ij}, f_0 \in [0, 1)$.

The **Gomory Fractional Cut**:
$$\sum_{j \in \mathcal{N}} f_{ij} x_j \ge f_0,$$
cuts off the fractional LP optimum without removing any valid integer points.

---

## 6. Convex Quadratic Programming (QP) & ADMM

> **Primary Source**:  
> Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto Bemporad, Stephen Boyd (2020). *"OSQP: An operator splitting solver for quadratic programs"*, **Mathematical Programming Computation**, 12: 637–672.  
> DOI: [10.1007/s12532-020-00179-2](https://doi.org/10.1007/s12532-020-00179-2)

### 6.1 Formulation

Convex Quadratic Program:
$$\begin{aligned}
\min_{\mathbf{x}} \quad & \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{l} \le A \mathbf{x} \le \mathbf{u},
\end{aligned}$$
where $P \in \mathbb{S}_+^n$ is symmetric positive semidefinite ($P \succeq 0$).

### 6.2 OSQP ADMM Formulation

Splitting variables via $\mathbf{z} = A \mathbf{x}$:
$$\min_{\mathbf{x}, \mathbf{z}} \frac{1}{2} \mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} + \mathcal{I}_{[\mathbf{l}, \mathbf{u}]}(\mathbf{z}) \quad \text{s.t.} \quad A \mathbf{x} - \mathbf{z} = \mathbf{0}.$$

ADMM Iterations:
1. **Solve Quasi-Definite Linear System**:
   $$\begin{bmatrix} P + \sigma I & A^T \\ A & -\rho^{-1} I \end{bmatrix} \begin{bmatrix} \mathbf{x}^{k+1} \\ \boldsymbol{\nu}^{k+1} \end{bmatrix} = \begin{bmatrix} \sigma \mathbf{x}^k - \mathbf{q} \\ \mathbf{z}^k - \rho^{-1} \mathbf{y}^k \end{bmatrix}.$$
   The coefficient matrix is symmetric quasi-definite: factorized **once** via sparse $LDL^T$ at iteration 0; every subsequent iteration requires only forward/backward triangular substitution ($O(\text{nnz})$).
2. **Project Slack Variable**:
   $$\mathbf{z}^{k+1} = \text{proj}_{[\mathbf{l}, \mathbf{u}]} \left( A \mathbf{x}^{k+1} + \rho^{-1} \mathbf{y}^k \right).$$
3. **Dual Multiplier Update**:
   $$\mathbf{y}^{k+1} = \mathbf{y}^k + \rho (A \mathbf{x}^{k+1} - \mathbf{z}^{k+1}).$$

---

## 7. Independent Verification & Exact Certificates

> **Primary Sources**:  
> 1. David L. Applegate, William Cook, Sanjeeb Dash, Daniel G. Espinoza (2007). *"Exact solutions to linear programming problems"*, **Operations Research Letters**, 35(4): 445–449.  
> 2. Kevin K. H. Cheung, Ambros Gleixner, Daniel E. Steffy (2017). *"Verifying integer programming results"*, **Mathematical Programming** / arXiv:1611.08832.

An optimization solver must not ask its user or an evaluator to "trust" its internal state. A bug in matrix updates or tolerance handling could silently return a non-optimal or infeasible point.

### 7.1 Farkas’ Lemma & Certificates of Infeasibility

#### Primal Infeasibility Certificate
For the system $A \mathbf{x} = \mathbf{b}$, $\mathbf{x} \ge \mathbf{0}$, by Farkas’ Lemma:
$$\text{Either } \exists \mathbf{x} \ge \mathbf{0} \text{ with } A \mathbf{x} = \mathbf{b}, \quad \text{or } \exists \mathbf{y} \in \mathbb{R}^m \text{ with } A^T \mathbf{y} \le \mathbf{0} \text{ and } \mathbf{b}^T \mathbf{y} > 0.$$
The vector $\mathbf{y}$ is a **Farkas ray**:
- In dual simplex, when no leaving row can find a valid entering column in CHUZC, the tableau row $\widehat{\mathbf{e}}_p = \mathbf{e}_p^T B^{-1}$ directly provides the Farkas ray.
- The independent verifier checks:
  1. $A^T \mathbf{y} \le \epsilon_{\text{tol}}$,
  2. $\mathbf{b}^T \mathbf{y} \ge \delta > 0$,
  proving primal infeasibility unconditionally.

#### Unboundedness Certificate (Primal Recession Ray)
If an LP is unbounded, there exists direction $\mathbf{d} \ge \mathbf{0}$ such that:
$$A \mathbf{d} = \mathbf{0}, \quad \mathbf{c}^T \mathbf{d} < 0.$$

### 7.2 Independent Verification Metrics

The verifier accepts solution $(\mathbf{x}, \mathbf{y})$ if all normalized residuals are within tolerance $\epsilon = 10^{-7}$:
1. **Primal Feasibility Residual**:
   $$\epsilon_P = \frac{\|A \mathbf{x} - \mathbf{b}\|_\infty}{1 + \|\mathbf{b}\|_\infty + \|A\|_\infty \|\mathbf{x}\|_\infty} \le \epsilon.$$
2. **Dual Feasibility Residual**:
   $$\epsilon_D = \frac{\|\mathbf{c} - A^T \mathbf{y} - \mathbf{s}\|_\infty}{1 + \|\mathbf{c}\|_\infty + \|A^T\|_\infty \|\mathbf{y}\|_\infty} \le \epsilon.$$
3. **Strong Duality Gap**:
   $$\epsilon_{\text{gap}} = \frac{|\mathbf{c}^T \mathbf{x} - \mathbf{b}^T \mathbf{y}|}{1 + |\mathbf{c}^T \mathbf{x}| + |\mathbf{b}^T \mathbf{y}|} \le \epsilon.$$

---

## 8. Refinery Domain Optimization (MRPL Operational Models)

> **Context**: Problem Statement SIH26119 is specified by **Mangalore Refinery and Petrochemicals Limited (MRPL)**, a 15 MMTPA coastal refinery operating complex Crude Distillation Units (CDU), Vacuum Distillation Units (VDU), Hydrocrackers, and CCR units.

### 8.1 CDU Crude Selection & Cut-Yield Modeling

A weekly planning LP decides the blend of crude oils $k \in \mathcal{C}$ (e.g., Arab Light, Maya, Bonny Light, Mumbai High) fed to CDU:
$$\max_{\mathbf{u}, \mathbf{x}} \sum_{p \in \mathcal{P}} \text{Price}_p \cdot \text{Production}_p - \sum_{k \in \mathcal{C}} \text{Cost}_k \cdot \text{Feed}_k - \text{OperatingCosts}.$$

Constraints:
1. **CDU Throughput Capacity**:
   $$\sum_{k \in \mathcal{C}} \text{Feed}_k \le \text{Capacity}_{\text{CDU}} \quad (\text{e.g., 40,000 bpd}).$$
2. **Volumetric Cut Yields**:
   $$\text{CutYield}_{p, k} \times \text{Feed}_k = \text{Stream}_{p, k}.$$
   Total cut production (Naphtha, Kerosene/ATF, Gasoil/Diesel, Vacuum Residue):
   $$\text{Yield}_p = \sum_{k \in \mathcal{C}} \alpha_{p, k} \text{Feed}_k.$$

### 8.2 Gasoline & Diesel Quality Blending (Sulfur Giveaway)

Refineries blend intermediate streams into commercial fuels meeting strict Bharat Stage VI (BS-VI) specifications:
1. **Maximum Sulfur Spec (10 ppm for BS-VI)**:
   $$\sum_{s \in \text{Streams}} \text{Sulfur}_s \cdot x_s \le 10.0 \times \sum_{s} x_s.$$
2. **Research Octane Number (RON $\ge 91$)**:
   $$\sum_{s} \text{RON}_s \cdot x_s \ge 91.0 \times \sum_{s} x_s.$$
3. **Reid Vapor Pressure (RVP $\le 60$ kPa)**:
   $$\sum_{s} (\text{RVP}_s)^{1.25} \cdot x_s \le (60)^{1.25} \times \sum_{s} x_s \quad (\text{linearized chevron index}).$$

### 8.3 Shadow Prices & Economic Interpretation

The dual multipliers $\boldsymbol{\pi}$ returned by the simplex solver have immediate physical and economic meaning to MRPL operations:
- $\pi_{\text{CDU}}$: **Marginal refinery throughput value** (\$/barrel). If CDU capacity expands by 1 barrel, net refinery profit increases by $\pi_{\text{CDU}}$.
- $\pi_{\text{Sulfur}}$: **Desulfurization marginal cost** (\$/ppm-barrel). The exact shadow cost of the BS-VI sulfur constraint. Reveals how much MRPL should pay for low-sulfur sweet crudes vs high-sulfur sour crudes.
- $\pi_{\text{Diesel\_Commit}}$: **Contractual opportunity cost** of meeting regional fuel supply commitments.

---

## 9. Numerical Integrity, IEEE 754 & Secure Engineering

> **Primary Source**:  
> Nicholas J. Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM.

### 9.1 Floating-Point Arithmetic Rules
- **NaN / Infinity Handling**: In IEEE 754, comparisons with NaN return `false` (except `!=`). Naive checks like `if (x > 0)` fail to catch NaNs. SIHOpt enforces `std::isfinite(val)` on every input, matrix entry, pivot, direction vector, and candidate solution.
- **Fail-Closed Principle**: If matrix factorization encounters a pivot $|\alpha| < 10^{-13}$, or if condition estimate exceeds $10^{14}$, the engine returns `SolveStatus::numerical_failure`. It NEVER reports `Optimal` or `Infeasible` when numerical integrity is compromised.

### 9.2 Secure Resource Bounds (CWE Mitigations)
- **CWE-190 / CWE-680 (Integer Overflow to Buffer Overrun)**:
  `checked_product(rows, columns)` uses 64-bit unsigned arithmetic with overflow detection before any heap allocation.
- **RFC 8259 Compliant Output**:
  All string serialization (model names, row identifiers, error messages) escapes all control characters ($< 0x20$) via `\u00XX` hexadecimal encoding to prevent JSON injection in judge/evaluation scripts.

---

## 10. Comprehensive Annotated Bibliography

1. **Huangfu, Q., & Hall, J. A. J. (2018)**. *Parallelizing the dual revised simplex method*. **Mathematical Programming Computation**, 10(1), 119–142. [doi:10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).  
   *Significance*: Foundational dual revised simplex paper; establishes DSE weight update, PAMI, and SIP.
2. **Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021)**. *Practical Large-Scale Linear Programming Using Primal-Dual Hybrid Gradient*. **arXiv:2106.04756**.  
   *Significance*: Blueprint for GPU first-order LP solvers; introduces PDLP with adaptive restarts and Ruiz scaling.
3. **Achterberg, T., Koch, T., & Martin, A. (2005)**. *Branching rules revisited*. **Operations Research Letters**, 33(1), 42–54. [doi:10.1016/j.orl.2004.04.007](https://doi.org/10.1016/j.orl.2004.04.007).  
   *Significance*: Establishes reliability branching as the gold standard for MILP variable selection.
4. **Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020)**. *OSQP: An operator splitting solver for quadratic programs*. **Mathematical Programming Computation**, 12(4), 637–672. [doi:10.1007/s12532-020-00179-2](https://doi.org/10.1007/s12532-020-00179-2).  
   *Significance*: State-of-the-art ADMM solver for convex QP; single initial $LDL^T$ factorization with fast substitutions.
5. **Andersen, E. D., & Andersen, K. D. (1995)**. *Presolving in linear programming*. **Mathematical Programming**, 71(2), 221–245. [doi:10.1007/BF01586000](https://doi.org/10.1007/BF01586000).  
   *Significance*: Classic reference for reversible LP reductions and exact postsolve variable reconstruction.
6. **Forrest, J. J., & Goldfarb, D. (1992)**. *Steepest-edge simplex algorithms for linear programming*. **Mathematical Programming**, 57(1), 341–374. [doi:10.1007/BF01581089](https://doi.org/10.1007/BF01581089).  
   *Significance*: Derives the $O(m)$ recurrence relation for dual and primal steepest-edge weights.
7. **Harris, P. M. (1973)**. *Pivot selection methods of the Devex LP code*. **Mathematical Programming**, 5(1), 1–28. [doi:10.1007/BF01580108](https://doi.org/10.1007/BF01580108).  
   *Significance*: Introduces the two-pass ratio test for avoiding near-zero pivots in simplex.
8. **Gilbert, J. R., & Peierls, T. (1988)**. *Sparse partial pivoting in time proportional to arithmetic operations*. **SIAM Journal on Scientific and Statistical Computing**, 9(5), 862–874. [doi:10.1137/0909058](https://doi.org/10.1137/0909058).  
   *Significance*: Graph-reach algorithm for sparse triangular solve execution in $O(\text{flops})$.
9. **Applegate, D. L., Cook, W., Dash, S., & Espinoza, D. G. (2007)**. *Exact solutions to linear programming problems*. **Operations Research Letters**, 35(4), 445–449. [doi:10.1016/j.orl.2006.12.010](https://doi.org/10.1016/j.orl.2006.12.010).  
   *Significance*: Independent mathematical certificates for LP optimality and feasibility.
10. **Bland, R. G. (1977)**. *New finite pivoting rules for the simplex method*. **Mathematics of Operations Research**, 2(2), 103–107. [doi:10.1287/moor.2.2.103](https://doi.org/10.1287/moor.2.2.103).  
    *Significance*: Proves cycle prevention via deterministic smallest-index entering and leaving rules.
11. **Higham, N. J. (2002)**. *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027).  
    *Significance*: Authoritative reference for backward error analysis, condition numbers, and pivot growth.
12. **Bixby, R. E., Fenelon, M., Gu, Z., Rothberg, E., & Wunderling, R. (2000)**. *MIP: Theory and Practice — Closing the Gap*. System Modelling and Optimization, 19–49.  
    *Significance*: Landmark overview of modern MILP technology: B&B, cuts, presolve, and dual warm-starts.
13. **Pinto, J. M., Joly, M., & Moro, L. F. L. (2000)**. *Planning and scheduling models for refinery operations*. **Computers & Chemical Engineering**, 24(9–10), 2259–2276. [doi:10.1016/S0098-1354(00)00588-4](https://doi.org/10.1016/S0098-1354(00)00588-4).  
    *Significance*: Operational linear programming models for refinery CDU/VDU scheduling and blending.
