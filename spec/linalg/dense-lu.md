# Dense LU contract

M2 factors square finite matrices with row partial pivoting, rejects pivots at or below a scaled tolerance, supports Ax=b and A^T x=b, and reports pivot diagnostics. Residuals are independently recomputed. This is a dense correctness oracle, not optimized production factorization.
