# NumericalPolicy — SPEC-M0-NUM-01

All values are versioned, serialized, finite, nonnegative, and scale-aware. Solver stopping tolerances are separate from final verification tolerances. No local epsilon is authoritative.

Proposed defaults pending review: primal abs/rel `1e-7/1e-7`; dual abs/rel `1e-7/1e-7`; complementarity `1e-7`; integrality `1e-6`; certificate `1e-8`; pivot abs/rel `1e-12/1e-10`; storage zero-drop `1e-14`; symmetry abs/rel `1e-12/1e-10`; PSD `1e-10 * matrix_scale`; refinement `1e-10`; objective abs/rel `1e-8/1e-8`; MILP gap abs/rel `1e-8/1e-4`; GPU LP target `1e-6`.

The final policy must also define scaling limits, condition warnings, and refactorization thresholds. Evidence reports configured and achieved values and the scale used by every failed comparison.
