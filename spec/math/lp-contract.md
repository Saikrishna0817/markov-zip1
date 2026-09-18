# LP contract — SPEC-M0-LP-01

The immutable original model is

\[\min_x c^T x+c_0\quad	ext{s.t.}\quad l_r\le Ax\le u_r,\;l_x\le x\le u_x.\]

Infinity is an extended-real bound state and is never approximated by a finite sentinel. Maximization is reversibly mapped to minimization by objective negation.

For finite bounds use nonnegative multipliers: `alpha` for `Ax-u_r<=0`, `beta` for `l_r-Ax<=0`, `gamma` for `x-u_x<=0`, and `delta` for `l_x-x<=0`. Multipliers for absent bounds equal zero.

Stationarity: `c + A^T(alpha-beta) + gamma-delta = 0`.

Dual objective: `c0 - u_r^T alpha + l_r^T beta - u_x^T gamma + l_x^T delta`.

Complementarity applies multiplier-by-slack to all finite row and variable bounds. Final verification uses the original model after reversible postsolve.
