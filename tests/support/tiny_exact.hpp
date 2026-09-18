#pragma once
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sihopt::test_support {
class Rational final {
  public:
    Rational(std::int64_t numerator = 0, std::int64_t denominator = 1) {
        if (denominator == 0) throw std::invalid_argument("zero denominator");
        if (numerator == std::numeric_limits<std::int64_t>::min() || denominator == std::numeric_limits<std::int64_t>::min()) throw std::overflow_error("tiny rational input outside safe range");
        if (numerator < -1000000 || numerator > 1000000 || denominator < -1000000 || denominator > 1000000) throw std::overflow_error("tiny rational input outside safe range");
        if (denominator < 0) { numerator = -numerator; denominator = -denominator; }
        const auto divisor = std::gcd(numerator, denominator);
        numerator_ = numerator / divisor; denominator_ = denominator / divisor;
    }
    [[nodiscard]] std::int64_t numerator() const noexcept { return numerator_; }
    [[nodiscard]] std::int64_t denominator() const noexcept { return denominator_; }
    friend bool operator==(const Rational&, const Rational&) = default;
    friend bool operator<(const Rational& a, const Rational& b) { return a.numerator_ * b.denominator_ < b.numerator_ * a.denominator_; }
    friend bool operator<=(const Rational& a, const Rational& b) { return !(b < a); }
    friend Rational operator+(const Rational& a, const Rational& b) {
        return Rational(a.numerator_ * b.denominator_ + b.numerator_ * a.denominator_, a.denominator_ * b.denominator_);
    }
    friend Rational operator*(const Rational& a, const Rational& b) {
        return Rational(a.numerator_ * b.numerator_, a.denominator_ * b.denominator_);
    }
  private:
    std::int64_t numerator_{};
    std::int64_t denominator_{1};
};

inline void enumerate_integer_box(const std::vector<int>& lower, const std::vector<int>& upper, const std::function<void(const std::vector<int>&)>& visit) {
    if (lower.size() != upper.size()) throw std::invalid_argument("box dimension mismatch");
    std::vector<int> point = lower;
    for (std::size_t i = 0; i < lower.size(); ++i) if (lower[i] > upper[i]) return;
    if (point.empty()) { visit(point); return; }
    while (true) {
        visit(point);
        std::size_t position = point.size();
        while (position > 0U && point[position - 1U] == upper[position - 1U]) --position;
        if (position == 0U) return;
        --position;
        ++point[position];
        for (std::size_t j = position + 1U; j < point.size(); ++j) point[j] = lower[j];
    }
}
}
