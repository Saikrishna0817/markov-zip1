#include "markov_cero/lp/dual/dual_simplex.hpp"
#include "markov_cero/verify/reference_lp_verifier.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace markov_cero;
namespace {void req(bool q,const char*m){if(!q)throw std::runtime_error(m);}transform::CanonicalModel make_model(double lo,double up){transform::CanonicalModel m;m.matrix={2,3,{-1,1,0,1,0,1}};m.rhs={-lo,up};m.objective={1,0,0};m.record.objective_sign=1;m.record.structural_variables=1;m.record.variables.resize(1);m.validate();return m;}}
int main(){auto seed=make_model(0,20);auto cold=lp::dual::solve(seed);auto basis=cold.basis_state;std::mt19937_64 rng(0x4d3450524f50ULL);std::uniform_real_distribution<double>d(0,20);for(int k=0;k<500;++k){double a=d(rng),b=d(rng);double lo=std::min(a,b),up=std::max(a,b);auto m=make_model(lo,up);auto warm=lp::dual::solve(m,{},basis);auto reference=lp::reference::solve(m);req(warm.solution.status==lp::reference::SolveStatus::optimal,"random warm status");req(reference.status==warm.solution.status,"random status parity");req(std::abs(reference.objective-warm.solution.objective)<1e-8,"random objective parity");req(verify::verify_reference_result(m,warm.solution).accepted,"random warm verification");lp::dual::Options strict;strict.harris_ratio=false;auto second=lp::dual::solve(m,strict,basis);req(second.solution.status==warm.solution.status&&std::abs(second.solution.objective-warm.solution.objective)<1e-8,"Harris strict parity");}
 for(int k=0;k<100;++k){double up=d(rng);auto m=make_model(up+1+d(rng),up);auto warm=lp::dual::solve(m,{},basis);req(warm.solution.status==lp::reference::SolveStatus::infeasible,"random infeasible status");req(verify::verify_reference_result(m,warm.solution).accepted,"random Farkas verification");}
 std::cout<<"M4 randomized warm-start tests passed\n";}
