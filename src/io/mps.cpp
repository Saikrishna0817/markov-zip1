#include "markov_cero/io/mps.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace markov_cero::io {
namespace {
enum class Section {
    none,
    objective_sense,
    objective_name,
    rows,
    columns,
    rhs,
    ranges,
    bounds,
    end
};
struct Row {
    char type;
    std::string name;
    double rhs{0.0};
    bool has_rhs{false};
    double range{0.0};
    bool has_range{false};
};
struct Column {
    std::string name;
    model::VariableType type{model::VariableType::continuous};
    model::Bound lower{model::Bound::finite(0.0)};
    model::Bound upper{model::Bound::positive_infinity()};
    double objective{0.0};
};
std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}
std::vector<std::string> tokens(const std::string& line) {
    std::istringstream in(line);
    std::vector<std::string> out;
    for (std::string value; in >> value;)
        out.push_back(value);
    return out;
}
std::string unquote(std::string value) {
    if (value.size() >= 2U && ((value.front() == '\'' && value.back() == '\'') ||
                               (value.front() == '"' && value.back() == '"')))
        return value.substr(1U, value.size() - 2U);
    return value;
}
double number(const std::string& text, std::size_t line) {
    std::string normalized = text;
    std::replace(normalized.begin(), normalized.end(), 'D', 'E');
    std::replace(normalized.begin(), normalized.end(), 'd', 'e');
    std::size_t used = 0U;
    double value = 0.0;
    try {
        value = std::stod(normalized, &used);
    } catch (...) {
        throw MpsError(line, "invalid numeric token: " + text);
    }
    if (used != normalized.size() || !std::isfinite(value))
        throw MpsError(line, "numeric token must be finite: " + text);
    return value;
}
bool header(const std::string& token) {
    static const std::vector<std::string> names{"NAME", "OBJSENSE", "OBJNAME", "ROWS",  "COLUMNS",
                                                "RHS",  "RANGES",   "BOUNDS",  "ENDATA"};
    return std::find(names.begin(), names.end(), token) != names.end();
}
} // namespace

MpsError::MpsError(std::size_t line, std::string message)
    : std::runtime_error("MPS line " + std::to_string(line) + ": " + std::move(message)),
      line_(line) {}
std::size_t MpsError::line() const noexcept { return line_; }

model::Model parse_mps(std::istream& input, const MpsLimits& limits) {
    if (limits.maximum_bytes == 0U || limits.maximum_lines == 0U || limits.maximum_name_bytes == 0U)
        throw std::invalid_argument("MPS limits must be positive");
    Section section = Section::none;
    std::string problem_name;
    std::string objective_name;
    model::ObjectiveSense sense = model::ObjectiveSense::minimize;
    std::vector<Row> rows;
    std::unordered_map<std::string, std::size_t> row_by_name;
    std::vector<Column> columns;
    std::unordered_map<std::string, std::size_t> column_by_name;
    struct Coefficient {
        std::size_t row;
        std::size_t column;
        double value;
    };
    std::vector<Coefficient> coefficients;
    std::string rhs_vector, range_vector, bound_vector;
    bool in_integer_block = false;
    bool saw_end = false;
    std::size_t bytes = 0U;
    std::size_t line_number = 0U;
    auto require_name = [&](const std::string& name) {
        if (name.empty() || name.size() > limits.maximum_name_bytes)
            throw MpsError(line_number, "invalid or oversized name");
        const bool printable_ascii = std::all_of(name.begin(), name.end(), [](unsigned char value) {
            return value >= 33U && value <= 126U;
        });
        if (!printable_ascii)
            throw MpsError(line_number, "names must use printable ASCII without spaces");
    };
    auto find_row = [&](const std::string& name) -> std::size_t {
        const auto it = row_by_name.find(name);
        if (it == row_by_name.end())
            throw MpsError(line_number, "unknown row: " + name);
        return it->second;
    };
    auto find_or_add_column = [&](const std::string& name) -> std::size_t {
        require_name(name);
        const auto found = column_by_name.find(name);
        if (found != column_by_name.end())
            return found->second;
        if (columns.size() >= limits.maximum_columns)
            throw MpsError(line_number, "column limit exceeded");
        const auto index = columns.size();
        Column column;
        column.name = name;
        if (in_integer_block) {
            column.type = model::VariableType::integer;
            column.upper = model::Bound::finite(1.0);
        }
        columns.push_back(column);
        column_by_name.emplace(name, index);
        return index;
    };

    for (std::string raw; std::getline(input, raw);) {
        if (line_number == std::numeric_limits<std::size_t>::max())
            throw MpsError(line_number, "line counter overflow");
        ++line_number;
        const std::size_t delimiter_bytes = input.eof() ? 0U : 1U;
        if (raw.size() > limits.maximum_bytes ||
            delimiter_bytes > limits.maximum_bytes - raw.size() ||
            bytes > limits.maximum_bytes - raw.size() - delimiter_bytes)
            throw MpsError(line_number, "byte limit exceeded");
        bytes += raw.size() + delimiter_bytes;
        if (line_number > limits.maximum_lines)
            throw MpsError(line_number, "line limit exceeded");
        if (!raw.empty() && raw.back() == '\r') {
            raw.pop_back();
        }
        const std::string line = trim(raw);
        if (line.empty() || line.front() == '*')
            continue;
        const auto fields = tokens(line);
        if (fields.empty())
            continue;
        const std::string first = fields.front();
        if (saw_end)
            throw MpsError(line_number, "content after ENDATA");
        const bool starts_in_col1 = (!raw.empty() && raw.front() != ' ' && raw.front() != '\t');
        if (starts_in_col1) {
            if (!header(first))
                throw MpsError(line_number, "unknown section: " + first);
            if (first != "NAME" && fields.size() != 1U)
                throw MpsError(line_number, "section header takes no values");
            if (first == "NAME") {
                section = Section::none;
                if (fields.size() >= 2U) {
                    require_name(fields[1]);
                    problem_name = fields[1];
                }
            } else if (first == "OBJSENSE")
                section = Section::objective_sense;
            else if (first == "OBJNAME")
                section = Section::objective_name;
            else if (first == "ROWS")
                section = Section::rows;
            else if (first == "COLUMNS")
                section = Section::columns;
            else if (first == "RHS")
                section = Section::rhs;
            else if (first == "RANGES")
                section = Section::ranges;
            else if (first == "BOUNDS")
                section = Section::bounds;
            else {
                section = Section::end;
                saw_end = true;
            }
            continue;
        }
        if (section == Section::objective_sense) {
            if (fields.size() != 1U)
                throw MpsError(line_number, "OBJSENSE requires one value");
            if (fields[0] == "MIN" || fields[0] == "MINIMIZE")
                sense = model::ObjectiveSense::minimize;
            else if (fields[0] == "MAX" || fields[0] == "MAXIMIZE")
                sense = model::ObjectiveSense::maximize;
            else {
                throw MpsError(line_number, "unknown objective sense");
            }
            section = Section::none;
            continue;
        }
        if (section == Section::objective_name) {
            if (fields.size() != 1U)
                throw MpsError(line_number, "OBJNAME requires one row name");
            objective_name = fields[0];
            section = Section::none;
            continue;
        }
        if (section == Section::rows) {
            if (fields.size() != 2U || fields[0].size() != 1U)
                throw MpsError(line_number, "ROWS record requires type and name");
            const char type = fields[0][0];
            if (type != 'N' && type != 'E' && type != 'L' && type != 'G')
                throw MpsError(line_number, "unsupported row type");
            require_name(fields[1]);
            if (row_by_name.contains(fields[1]))
                throw MpsError(line_number, "duplicate row name: " + fields[1]);
            if (rows.size() >= limits.maximum_rows)
                throw MpsError(line_number, "row limit exceeded");
            row_by_name.emplace(fields[1], rows.size());
            rows.push_back({type, fields[1]});
            if (type == 'N' && objective_name.empty())
                objective_name = fields[1];
            continue;
        }
        if (section == Section::columns) {
            if (fields.size() == 3U && unquote(fields[1]) == "MARKER") {
                const auto marker = unquote(fields[2]);
                if (marker == "INTORG") {
                    if (in_integer_block)
                        throw MpsError(line_number, "nested INTORG");
                    in_integer_block = true;
                } else if (marker == "INTEND") {
                    if (!in_integer_block)
                        throw MpsError(line_number, "INTEND without INTORG");
                    in_integer_block = false;
                } else {
                    throw MpsError(line_number, "unknown MARKER value");
                }
                continue;
            }
            if (fields.size() != 3U && fields.size() != 5U)
                throw MpsError(line_number, "COLUMNS record requires one or two row/value pairs");
            const auto column = find_or_add_column(fields[0]);
            if (in_integer_block && columns[column].type == model::VariableType::continuous) {
                columns[column].type = model::VariableType::integer;
                columns[column].upper = model::Bound::finite(1.0);
            }
            for (std::size_t p = 1U; p < fields.size(); p += 2U) {
                const auto row = find_row(fields[p]);
                const double value = number(fields[p + 1U], line_number);
                if (rows[row].type == 'N') {
                    if (rows[row].name != objective_name)
                        throw MpsError(line_number, "coefficient references non-objective N row");
                    columns[column].objective += value;
                    if (!std::isfinite(columns[column].objective))
                        throw MpsError(line_number, "objective coefficient overflow");
                } else {
                    if (coefficients.size() >= limits.maximum_nonzeros)
                        throw MpsError(line_number, "nonzero limit exceeded");
                    coefficients.push_back({row, column, value});
                }
            }
            continue;
        }
        if (section == Section::rhs || section == Section::ranges) {
            std::size_t start_p = 1U;
            if (fields.size() == 2U || fields.size() == 4U) {
                start_p = 0U;
            } else if (fields.size() != 3U && fields.size() != 5U) {
                throw MpsError(line_number,
                               "RHS/RANGES record requires vector and row/value pairs");
            }
            if (start_p == 1U) {
                auto& selected = section == Section::rhs ? rhs_vector : range_vector;
                if (selected.empty())
                    selected = fields[0];
                else if (selected != fields[0])
                    throw MpsError(line_number, "multiple rim vectors are unsupported");
            }
            for (std::size_t p = start_p; p < fields.size(); p += 2U) {
                const auto row = find_row(fields[p]);
                if (rows[row].type == 'N') {
                    throw MpsError(line_number, "objective-row RHS/RANGES values are unsupported");
                }
                const double value = number(fields[p + 1U], line_number);
                if (section == Section::rhs) {
                    if (rows[row].has_rhs)
                        throw MpsError(line_number, "duplicate RHS row");
                    rows[row].rhs = value;
                    rows[row].has_rhs = true;
                } else {
                    if (rows[row].has_range)
                        throw MpsError(line_number, "duplicate RANGES row");
                    rows[row].range = value;
                    rows[row].has_range = true;
                }
            }
            continue;
        }
        if (section == Section::bounds) {
            if (fields.size() != 3U && fields.size() != 4U)
                throw MpsError(line_number, "BOUNDS record has invalid field count");
            const auto type = fields[0];
            if (bound_vector.empty())
                bound_vector = fields[1];
            else if (bound_vector != fields[1])
                throw MpsError(line_number, "multiple bound vectors are unsupported");
            const auto column = find_or_add_column(fields[2]);
            auto& value = columns[column];
            const bool needs_number =
                type == "LO" || type == "UP" || type == "FX" || type == "LI" || type == "UI";
            if (needs_number != (fields.size() == 4U))
                throw MpsError(line_number, "bound type has wrong value count");
            const double bound = needs_number ? number(fields[3], line_number) : 0.0;
            if (type == "LO")
                value.lower = model::Bound::finite(bound);
            else if (type == "UP")
                value.upper = model::Bound::finite(bound);
            else if (type == "FX")
                value.lower = value.upper = model::Bound::finite(bound);
            else if (type == "FR") {
                value.lower = model::Bound::negative_infinity();
                value.upper = model::Bound::positive_infinity();
            } else if (type == "MI")
                value.lower = model::Bound::negative_infinity();
            else if (type == "PL")
                value.upper = model::Bound::positive_infinity();
            else if (type == "BV") {
                value.type = model::VariableType::binary;
                value.lower = model::Bound::finite(0.0);
                value.upper = model::Bound::finite(1.0);
            } else if (type == "LI") {
                value.type = model::VariableType::integer;
                value.lower = model::Bound::finite(bound);
            } else if (type == "UI") {
                value.type = model::VariableType::integer;
                value.upper = model::Bound::finite(bound);
            } else
                throw MpsError(line_number, "unsupported bound type: " + type);
            continue;
        }
        throw MpsError(line_number, "record outside a supported section");
    }
    if (!saw_end)
        throw MpsError(line_number, "missing ENDATA");
    if (in_integer_block)
        throw MpsError(line_number, "missing INTEND");
    if (objective_name.empty() || !row_by_name.contains(objective_name))
        throw MpsError(line_number, "missing objective N row");
    std::vector<std::size_t> row_map(rows.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::string> row_names;
    std::vector<model::Bound> row_lower, row_upper;
    for (std::size_t old = 0; old < rows.size(); ++old) {
        const auto& row = rows[old];
        if (row.type == 'N')
            continue;
        row_map[old] = row_names.size();
        row_names.push_back(row.name);
        const double rhs = row.rhs;
        model::Bound lower = model::Bound::negative_infinity(),
                     upper = model::Bound::positive_infinity();
        if (row.type == 'E')
            lower = upper = model::Bound::finite(rhs);
        else if (row.type == 'L')
            upper = model::Bound::finite(rhs);
        else
            lower = model::Bound::finite(rhs);
        if (row.has_range) {
            const double width = std::abs(row.range);
            if (row.type == 'L')
                lower = model::Bound::finite(rhs - width);
            else if (row.type == 'G')
                upper = model::Bound::finite(rhs + width);
            else if (row.type == 'E') {
                if (row.range >= 0.0)
                    upper = model::Bound::finite(rhs + width);
                else
                    lower = model::Bound::finite(rhs - width);
            }
        }
        row_lower.push_back(lower);
        row_upper.push_back(upper);
    }
    model::SparseMatrixBuilder builder(row_names.size(), columns.size());
    for (const auto& entry : coefficients)
        builder.add(row_map[entry.row], entry.column, entry.value);
    model::Model result;
    result.name = problem_name;
    result.objective_sense = sense;
    result.matrix = builder.build();
    result.row_name = std::move(row_names);
    result.row_lower = std::move(row_lower);
    result.row_upper = std::move(row_upper);
    for (const auto& column : columns) {
        result.variable_name.push_back(column.name);
        result.objective.push_back(column.objective);
        result.variable_lower.push_back(column.lower);
        result.variable_upper.push_back(column.upper);
        result.variable_type.push_back(column.type);
    }
    result.validate();
    return result;
}
model::Model parse_mps_string(std::string_view input, const MpsLimits& limits) {
    if (input.size() > limits.maximum_bytes)
        throw MpsError(0U, "byte limit exceeded");
    std::istringstream stream{std::string(input)};
    return parse_mps(stream, limits);
}
} // namespace markov_cero::io
