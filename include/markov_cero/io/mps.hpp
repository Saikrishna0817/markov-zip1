#pragma once

#include "markov_cero/model/model.hpp"

#include <cstddef>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace markov_cero::io {

struct MpsLimits final {
    std::size_t maximum_bytes{16U * 1024U * 1024U};
    std::size_t maximum_lines{1'000'000U};
    std::size_t maximum_rows{1'000'000U};
    std::size_t maximum_columns{1'000'000U};
    std::size_t maximum_nonzeros{20'000'000U};
    std::size_t maximum_name_bytes{255U};
};

class MpsError final : public std::runtime_error {
  public:
    MpsError(std::size_t line, std::string message);
    [[nodiscard]] std::size_t line() const noexcept;

  private:
    std::size_t line_;
};

[[nodiscard]] model::Model parse_mps(std::istream& input, const MpsLimits& limits = {});
[[nodiscard]] model::Model parse_mps_string(std::string_view input, const MpsLimits& limits = {});

} // namespace markov_cero::io
