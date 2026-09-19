#include "markov_cero/io/mps.hpp"
#include "support/tiny_exact.hpp"
#include <stdexcept>
#include <cfenv>
#include <cmath>
#include "markov_cero/verify/primal_verifier.hpp"
#include <string>
#include <vector>

static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    using markov_cero::test_support::Rational;
    require(Rational(1, 2) + Rational(1, 3) == Rational(5, 6), "exact addition");
    require(Rational(2, 3) * Rational(9, 4) == Rational(3, 2), "exact multiplication");
    int count = 0; markov_cero::test_support::enumerate_integer_box({0, 1}, {2, 2}, [&](const std::vector<int>&){ ++count; }); require(count == 6, "box enumeration");
    const std::string ranged = "NAME R\r\nROWS\r\n N O\r\n L RL\r\n G RG\r\n E RE\r\nCOLUMNS\r\n X O 1 RL 1\r\n X RG 1 RE 1\r\nRHS\r\n R1 RL 10 RG 4\r\n R1 RE 7\r\nRANGES\r\n G1 RL 3 RG -2\r\n G1 RE -5\r\nBOUNDS\r\n FR B X\r\nENDATA\r\n";
    const auto model = markov_cero::io::parse_mps_string(ranged);
    require(model.row_lower[0].value == 7 && model.row_upper[0].value == 10, "L range");
    require(model.row_lower[1].value == 4 && model.row_upper[1].value == 6, "G range");
    require(model.row_lower[2].value == 2 && model.row_upper[2].value == 7, "E negative range");
    bool rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\nQSECTION\nENDATA\n"); } catch (const markov_cero::io::MpsError&) { rejected=true; } require(rejected,"unknown section");
    markov_cero::io::MpsLimits limits; limits.maximum_bytes=10; rejected=false; try { (void)markov_cero::io::parse_mps_string(ranged,limits); } catch(const markov_cero::io::MpsError&){rejected=true;} require(rejected,"byte limit");
    const auto integer = markov_cero::io::parse_mps_string("NAME I\nROWS\n N O\n L R\nCOLUMNS\n M 'MARKER' 'INTORG'\n X O 1 R 1\n M2 'MARKER' 'INTEND'\nRHS\n R1 R 1\nENDATA\n");
    require(integer.variable_type[0] == markov_cero::model::VariableType::integer, "integer marker");
    rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\nENDATA\nROWS\n"); } catch(const markov_cero::io::MpsError&){rejected=true;} require(rejected,"content after ENDATA");
    rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\n L R\nCOLUMNS\n X O 1 R 1\nRHS\n R1 O 3\nENDATA\n"); } catch(const markov_cero::io::MpsError&){rejected=true;} require(rejected,"objective RHS ambiguity");
    rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\n L R\nCOLUMNS\n Xé O 1 R 1\nENDATA\n"); } catch(const markov_cero::io::MpsError&){rejected=true;} require(rejected,"non-ASCII name");
    rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS junk\n N O\nENDATA\n"); } catch(const markov_cero::io::MpsError&){rejected=true;} require(rejected,"section header arity");
    rejected=false; try { (void)markov_cero::io::parse_mps_string("NAME X\nROWS\n N O\n L R\nCOLUMNS\n X O 1 R 1\nBOUNDS\n BV B X\n FR B X\nENDATA\n"); } catch(const std::invalid_argument&){rejected=true;} require(rejected,"binary bounds remain within zero and one");
    const int old_round=std::fegetround(); std::fesetround(FE_UPWARD);
    markov_cero::model::Model binary; binary.name="B"; binary.matrix=markov_cero::model::SparseMatrixBuilder(0,1).build(); binary.variable_name={"x"}; binary.objective={1}; binary.variable_lower={markov_cero::model::Bound::finite(0)}; binary.variable_upper={markov_cero::model::Bound::finite(1)}; binary.variable_type={markov_cero::model::VariableType::binary}; binary.validate();
    const auto rounded=markov_cero::verify::verify_primal(binary,{{1.2},1.2},{0,0},{0,0},0.0); require(std::abs(rounded.maximum_integrality_violation-0.2)<1e-12,"rounding-mode independent integrality"); std::fesetround(old_round);
    const auto tiny = markov_cero::io::parse_mps_string("NAME T\nROWS\n N O\n E BAND\nCOLUMNS\n X O 1 BAND 1\n Y O 2 BAND 1\nRHS\n R BAND 2\nRANGES\n G BAND 1\nBOUNDS\n UI B X 2\n UI B Y 2\nENDATA\n");
    int feasible_count=0, infeasible_count=0;
    markov_cero::test_support::enumerate_integer_box({0,0},{2,2},[&](const std::vector<int>& point){
        const Rational activity=Rational(point[0])+Rational(point[1]); const Rational objective=Rational(point[0])+Rational(2)*Rational(point[1]);
        const bool exact_feasible=Rational(2)<=activity && activity<=Rational(3);
        const auto checked=markov_cero::verify::verify_primal(tiny,{{static_cast<double>(point[0]),static_cast<double>(point[1])},static_cast<double>(objective.numerator())/objective.denominator()},{0,0},{0,0},0.0);
        require(checked.passed==exact_feasible,"exact exhaustive oracle/verifier disagreement"); if(exact_feasible) ++feasible_count; else ++infeasible_count;
    });
    require(feasible_count==5 && infeasible_count==4,"exact ranged-row enumeration counts");
    return 0;
}
