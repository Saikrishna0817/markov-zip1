# Sparse basis normative contract

1. Input is finite canonical square CSC with increasing row indices and no explicit zeros.
2. Factorization represents `P B=L U` and must reject a pivot at or below the configured singular tolerance.
3. `solve` and `solve_transpose` must reject dimension/non-finite input and non-finite output.
4. Factor nonzeros must never exceed the configured ceiling.
5. Column update `p<-a` records `d=B^-1 a` and must reject `|d_p|` at or below update pivot tolerance.
6. FTRAN applies updates chronologically; BTRAN applies transpose updates in reverse chronological order.
7. Count or density triggers require refactorization before claiming an unbounded update chain.
8. No successful solve may rely on DenseLu inside the sparse implementation.
9. The warm dual path must retain independent final-result verification and explicit cold fallback.
