# M4 mathematics and verification

The dual simplex starts from reduced costs r=c-A^T y that are nonnegative while some basic values xB=B^-1b may be negative. A negative basic variable leaves. If pi=B^-T e_p, then alpha=A^T pi is the leaving tableau row. An eligible entering nonbasic variable has alpha_j<0 and dual ratio r_j/(-alpha_j).

If no eligible column exists, -pi satisfies A^T(-pi)<=0 and b^T(-pi)=-xB_p>0, giving a Farkas infeasibility certificate. Otherwise a pivot repairs primal infeasibility while retaining dual feasibility. Conservative Harris selection uses a tolerance-expanded first pass and a largest-pivot second pass; final witness verification prevents a relaxed selection from becoming an unsupported success claim.

Cold/warm equivalence, strict and Harris ratio modes, steepest-edge and Bland pricing, feasible and infeasible RHS changes, stale/duplicate basis rejection, serialization corruption, iteration limits, randomized objective parity, and Farkas witnesses are tested.

Review remediation covers the valid zero-row warm basis, separates option failures from warm-basis fallback, includes objective offsets in telemetry, and rejects semantically invalid standalone basis artifacts during serialization and parsing. Model-dependent fingerprint and nonsingularity validation remains in `solve`.
