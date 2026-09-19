# Qualification refinery slice

Four MPS files exercise the user-facing solver. Answers are **computed** by `sihopt-solve`; they are not baked into the demo script.

| File | Expected typed outcome |
|---|---|
| `refinery-feasible.mps` | Optimal, independently verified |
| `refinery-infeasible.mps` | Infeasible with a checked Farkas witness |
| `refinery-malformed.mps` | InvalidModel |
| `refinery-limited.mps` | IterationLimit when run with `--iteration-limit 1` |

```sh
cmake -S ../.. -B ../../build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ../../build -j --target sihopt-solve
../../build/sihopt-solve refinery-feasible.mps
```

From the repository root, `bash run-qualification-demo.sh` runs the feasible case end to end.
