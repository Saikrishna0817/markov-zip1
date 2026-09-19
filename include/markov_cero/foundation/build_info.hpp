#pragma once
#include <string_view>
namespace markov_cero::foundation {
[[nodiscard]] constexpr std::string_view version() noexcept { return "0.5.1"; }
[[nodiscard]] constexpr std::string_view milestone() noexcept { return "M5"; }
[[nodiscard]] constexpr bool contains_solver_algorithms() noexcept { return true; }
}
