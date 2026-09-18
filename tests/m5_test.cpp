#include "sihopt/linalg/dense_lu.hpp"
#include "sihopt/linalg/sparse_basis.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace sihopt::linalg;
namespace {
void req(bool q,const char*m){if(!q)throw std::runtime_error(m);}
DenseMatrix dense_from_columns(const std::vector<std::vector<double>>&c){DenseMatrix a{c.size(),c.size(),std::vector<double>(c.size()*c.size())};for(std::size_t j=0;j<c.size();++j)for(std::size_t i=0;i<c.size();++i)a.values[i*c.size()+j]=c[j][i];a.validate();return a;}
double error(const std::vector<double>&a,const std::vector<double>&b){double e=0;for(std::size_t i=0;i<a.size();++i)e=std::max(e,std::abs(a[i]-b[i]));return e;}
}
int main(){
 std::vector<std::vector<double>>cols{{4,1,0},{1,5,1},{0,2,6}};auto csc=SparseCsc::from_columns(3,cols);auto dense=dense_from_columns(cols);auto sparse=SparseLu::factorize(csc);auto dlu=DenseLu::factorize(dense);std::vector<double>b{7,-2,4};auto x=sparse.solve(b);auto xd=dlu.solve(b);req(error(x,xd)<1e-12,"sparse FTRAN parity");req(sparse_infinity_residual(csc,x,b)<1e-12,"sparse residual");auto xt=sparse.solve_transpose(b);auto xdt=dlu.solve_transpose(b);req(error(xt,xdt)<1e-12,"sparse BTRAN parity");req(sparse_infinity_residual(csc,xt,b,true)<1e-12,"sparse transpose residual");req(sparse.diagnostics().factor_nonzeros>=csc.values.size(),"factor metrics");
 SparseBasisOptions options;options.maximum_updates=3;options.eta_density_trigger=1;auto basis=SparseBasisFactorization::factorize(csc,options);std::vector<double>replacement{2,1,1};basis.replace_column(1,replacement);cols[1]=replacement;auto updated_dense=dense_from_columns(cols);auto updated_lu=DenseLu::factorize(updated_dense);x=basis.solve(b);req(error(x,updated_lu.solve(b))<1e-11,"eta FTRAN parity");xt=basis.solve_transpose(b);req(error(xt,updated_lu.solve_transpose(b))<1e-11,"eta BTRAN parity");req(basis.statistics().updates==1&&basis.statistics().current_update_chain==1,"eta statistics");basis.replace_column(1,std::vector<double>{1,3,1});cols[1]={1,3,1};basis.replace_column(1,std::vector<double>{1,4,2});cols[1]={1,4,2};req(basis.needs_refactorization(),"update trigger");basis.refactorize();req(!basis.needs_refactorization()&&basis.statistics().current_update_chain==0&&basis.statistics().refactorizations==2,"refactor reset");updated_dense=dense_from_columns(cols);updated_lu=DenseLu::factorize(updated_dense);req(error(basis.solve(b),updated_lu.solve(b))<1e-11,"post-refactor parity");
 bool threw=false;try{(void)SparseCsc{2,2,{0,2,2},{1,0},{1,2}}.dense_column(0);}catch(const std::invalid_argument&){threw=true;}req(threw,"unsorted CSC rejected");threw=false;try{(void)SparseCsc::from_columns(2,{{1,0},{0,std::numeric_limits<double>::infinity()}});}catch(const std::invalid_argument&){threw=true;}req(threw,"non-finite sparse value rejected");threw=false;try{(void)SparseLu::factorize(SparseCsc::from_columns(2,{{1,2},{2,4}}));}catch(const std::runtime_error&){threw=true;}req(threw,"singular sparse basis rejected");
 std::mt19937_64 rng(0x4d35535041525345ULL);std::uniform_real_distribution<double>v(-1,1);for(int trial=0;trial<250;++trial){const std::size_t n=1+static_cast<std::size_t>(rng()%18);std::vector<std::vector<double>>a(n,std::vector<double>(n));for(std::size_t j=0;j<n;++j)for(std::size_t i=0;i<n;++i)if((rng()%5)==0)a[j][i]=v(rng);for(std::size_t i=0;i<n;++i)a[i][i]+=static_cast<double>(n)+2;auto sm=SparseCsc::from_columns(n,a);auto sl=SparseLu::factorize(sm);auto dm=dense_from_columns(a);auto dl=DenseLu::factorize(dm);std::vector<double>rhs(n);for(auto&q:rhs)q=v(rng);req(error(sl.solve(rhs),dl.solve(rhs))<1e-9,"random FTRAN parity");req(error(sl.solve_transpose(rhs),dl.solve_transpose(rhs))<1e-9,"random BTRAN parity");}
 std::cout<<"M5 sparse basis tests passed\n";
}
