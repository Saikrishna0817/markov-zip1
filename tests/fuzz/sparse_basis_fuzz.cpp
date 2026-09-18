#include "sihopt/linalg/sparse_basis.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
using namespace sihopt::linalg;
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t*data,std::size_t size){if(size<2)return 0;const std::size_t n=1U+(data[0]%24U);std::vector<std::vector<double>>columns(n,std::vector<double>(n));std::size_t cursor=1;for(std::size_t j=0;j<n&&cursor<size;++j)for(std::size_t i=0;i<n&&cursor<size;++i){const std::int8_t raw=static_cast<std::int8_t>(data[cursor++]);if((raw&3)==0)columns[j][i]=static_cast<double>(raw)/32.0;}for(std::size_t i=0;i<n;++i)columns[i][i]+=static_cast<double>(n)+1.0;try{auto csc=SparseCsc::from_columns(n,columns);auto factor=SparseBasisFactorization::factorize(csc);std::vector<double>rhs(n);for(std::size_t i=0;i<n&&cursor<size;++i)rhs[i]=static_cast<double>(static_cast<std::int8_t>(data[cursor++]))/16.0;(void)factor.solve(rhs);(void)factor.solve_transpose(rhs);if(cursor<size){const std::size_t p=data[cursor++]%n;auto replacement=csc.dense_column(p);if(cursor<size)replacement[data[cursor++]%n]+=0.25;factor.replace_column(p,replacement);(void)factor.solve(rhs);(void)factor.solve_transpose(rhs);}}catch(const std::exception&){}return 0;}
