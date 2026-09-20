# Mathematical Formulations & Verification Theory — markov-cero

Formal mathematical foundation of markov-cero continuous and mixed-integer optimization algorithms.

---

## 1. Canonical Form & Duality Theory

### Canonical Form
Every linear program parsed by markov-cero is transformed into canonical standard equality form:
$$\min_{x} c^T x \quad \text{subject to} \quad A x = b, \quad x \ge 0$$
where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, and $c \in \mathbb{R}^n$.
Free variables are split into positive and negative parts ($x = x^+ - x^-$), and inequality
constraints are converted via non-negative slack and surplus variables.

### Dual Problem
The associated canonical dual problem is:
$$\max_{y, s} b^T y \quad \text{subject to} \quad A^T y + s = c, \quad s \ge 0$$
where $y \in \mathbb{R}^m$ are dual multipliers and $s \in \mathbb{R}^n$ are reduced costs.

### Karush-Kuhn-Tucker (KKT) Optimality Conditions
A primal-dual pair $(x^*, y^*, s^*)$ is optimal if and only if it satisfies:
1. **Primal Feasibility**: $A x^* = b, \quad x^* \ge 0$.
2. **Dual Feasibility**: $A^T y^* + s^* = c, \quad s^* \ge 0$.
3. **Complementary Slackness**: $x_j^* s_j^* = 0 \quad \forall j \in \{1, \dots, n\}$.
4. **Strong Duality**: $c^T x^* = b^T y^*$.

---

## 2. Infeasibility & Unboundedness Certificates

### Farkas' Lemma (Infeasibility Certificate)
If a primal linear program is infeasible, there exists a certificate vector $y \in \mathbb{R}^m$
satisfying:
$$A^T y \le 0 \quad \text{and} \quad b^T y > 0$$
The dual revised simplex engine emits this vector upon declaring primal infeasibility.

### Primal Ray (Unboundedness Certificate)
If a primal problem is unbounded, there exists a direction vector $d \in \mathbb{R}^n$ such that:
$$A d = 0, \quad d \ge 0, \quad \text{and} \quad c^T d < 0$$
Any point $x + \alpha d$ for $\alpha > 0$ remains feasible while driving the objective to $-\infty$.

---

## 3. Linear Algebra & Sparse Basis Updates

### Basis Representation
Let $B = [A_{\cdot B_1}, \dots, A_{\cdot B_m}]$ denote the $m \times m$ basis matrix. The basis is
factored as:
$$P B Q = L U$$
where $P, Q$ are permutation matrices, $L$ is unit lower triangular, and $U$ is upper triangular.

### Product-Form Eta Column Updates
When column $p$ leaves the basis and entering column $a_q$ enters, the updated basis is:
$$B_{\text{new}} = B_{\text{old}} E_k$$
where $E_k = I + (\eta_k - e_p) e_p^T$ is an elementary Eta matrix with column vector
$\eta_k = B_{\text{old}}^{-1} a_q$.
- **FTRAN**: Forward solve $B x = b$ via $L, U$ followed by chronological Eta application.
- **BTRAN**: Backward solve $B^T y = c_B$ via reverse-order Eta application followed by $U^T, L^T$.

---

## 4. Mixed-Integer Cutting Planes

### Gomory Mixed-Integer (GMI) Cuts
For a basic integer variable with fractional value $f_0 = x_i - \lfloor x_i \rfloor > 0$ from
tableau row:
$$x_i + \sum_{j \in N} \bar{a}_{ij} x_j = \bar{b}_i$$
the valid Gomory mixed-integer cut is:
$$\sum_{j \in N: f_j \le f_0} \frac{f_j}{f_0} x_j + \sum_{j \in N: f_j > f_0} \frac{1 - f_j}{1 - f_0} x_j \ge 1$$
where $f_j = \bar{a}_{ij} - \lfloor \bar{a}_{ij} \rfloor$.

### Mixed-Integer Rounding (MIR) Cuts
For a base inequality $\sum a_j x_j \le b$, the MIR function gives:
$$\sum \left( \lfloor a_j \rfloor + \frac{\max(0, a_j - \lfloor a_j \rfloor - (b - \lfloor b \rfloor))}{1 - (b - \lfloor b \rfloor)} \right) x_j \le \lfloor b \rfloor$$

---

## 5. Zero-Trust Verification Tolerances

The independent verifiers check solutions against the following scale-aware tolerances:
- **Primal Residual**: $\|A x - b\|_\infty \le \epsilon_{\text{feas}} (1 + \|b\|_\infty)$ with $\epsilon_{\text{feas}} = 10^{-7}$.
- **Dual Residual**: $\|A^T y + s - c\|_\infty \le \epsilon_{\text{dual}} (1 + \|c\|_\infty)$ with $\epsilon_{\text{dual}} = 10^{-7}$.
- **Complementarity**: $|c^T x - b^T y| \le \epsilon_{\text{opt}} (1 + |c^T x|)$ with $\epsilon_{\text{opt}} = 10^{-7}$.
- **Integrality**: $\max_{j \in I} |x_j - \text{round}(x_j)| \le 10^{-5}$.
