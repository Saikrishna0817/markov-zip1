# Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/sihopt-solve examples/blend.mps
bash run-qualification-demo.sh
ctest --test-dir build --output-on-failure
```

`sihopt-solve` prints one JSON object to stdout and a one-line verdict on stderr. Exit 0 is verified Optimal. Other typed statuses use nonzero codes (see `CAPABILITY-MATRIX.md`).
