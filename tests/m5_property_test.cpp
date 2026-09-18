#include "sihopt/linalg/dense_lu.hpp"
#include "sihopt/linalg/sparse_basis.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace sihopt::linalg;
namespace {
void req(bool v,const char*m){if(!v)throw std::runtime_error(m);}
DenseMatrix dense(const SparseCsc&s){DenseMatrix a{s.rows,s.columns,std::vector<double>(s.rows*s.columns)};for(std::size_t j=0;j<s.columns;++j)for(std::size_t p=s.column_offsets[j];p<s.column_offsets[j+1];++p)a.values[s.row_indices[p]*s.columns+j]=s.values[p];return a;}
double err(const std::vector<double>&a,const std::vector<double>&b){double e=0;for(std::size_t i=0;i<a.size();++i)e=std::max(e,std::abs(a[i]-b[i]));return e;}
}
int main(){std::mt19937_64 rng(0x4d3550524f504552ULL);std::uniform_real_distribution<double>value(-0.75,0.75);std::size_t accepted_updates=0;for(int trial=0;trial<400;++trial){const std::size_t n=2+static_cast<std::size_t>(rng()%11);std::vector<std::vector<double>>columns(n,std::vector<double>(n));for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<n;++i)if((rng()%4)==0)columns[j][i]=value(rng);for(std::size_t i=0;i<n;++i)columns[i][i]+=static_cast<double>(n)+1;SparseBasisOptions o;o.maximum_updates=7;o.eta_density_trigger=1;auto factor=SparseBasisFactorization::factorize(SparseCsc::from_columns(n,columns),o);for(int update=0;update<6;++update){const std::size_t p=static_cast<std::size_t>(rng()%n);auto replacement=factor.current_basis().dense_column(p);for(std::size_t i=0;i<n;++i)if((rng()%3)==0)replacement[i]+=0.15*value(rng);try{factor.replace_column(p,replacement);}catch(const std::runtime_error&){continue;}++accepted_updates;for(int rhs_index=0;rhs_index<3;++rhs_index){std::vector<double>rhs(n);for(auto&x:rhs)x=(rng()%3)==0?0:value(rng);auto dm=dense(factor.current_basis());auto oracle=DenseLu::factorize(dm);auto fresh=SparseLu::factorize(factor.current_basis());req(err(factor.solve(rhs),oracle.solve(rhs))<2e-9,"eta-chain FTRAN differential failure");req(err(factor.solve_transpose(rhs),oracle.solve_transpose(rhs))<2e-9,"eta-chain BTRAN differential failure");req(err(fresh.solve(rhs),oracle.solve(rhs))<2e-9,"fresh sparse FTRAN differential failure");req(err(fresh.solve_transpose(rhs),oracle.solve_transpose(rhs))<2e-9,"fresh sparse BTRAN differential failure");}if(factor.needs_refactorization())factor.refactorize();}}
req(accepted_updates>1500,"insufficient update coverage");std::vector<std::vector<double>>pivot_columns{{0,1,2},{2,1,0},{1,0,1}};auto pivot=SparseCsc::from_columns(3,pivot_columns);auto sl=SparseLu::factorize(pivot);auto dl=DenseLu::factorize(dense(pivot));std::vector<double>b{1,-2,3};req(err(sl.solve(b),dl.solve(b))<1e-11,"multi-row-pivot FTRAN failure");req(err(sl.solve_transpose(b),dl.solve_transpose(b))<1e-11,"multi-row-pivot BTRAN failure");std::cout<<"M5 randomized sparse/update properties passed: "<<accepted_updates<<" updates\n";}
