# Building

Requirements: CMake 3.25+, a C++20 compiler, Python 3.10+, and POSIX shell utilities.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Use `gcc-debug`, `gcc-release`, `clang-debug`, or `clang-release` presets when the matching compiler is installed. Unix Makefiles are the baseline generator; Ninja may be selected explicitly when installed. CUDA is neither required nor detected in M0.
