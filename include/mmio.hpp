#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <tuple>


// Minimal Matrix Market reader for coordinate real/general matrices.
// Supports 1-based or 0-based indices; converts to 0-based.
struct COO {
int M{0}, N{0};
std::vector<int> rows, cols;
std::vector<float> vals;
};


inline COO read_matrix_market_coo(const std::string &path) {
std::ifstream in(path);
if(!in) throw std::runtime_error("Failed to open MTX: " + path);
std::string line;
bool header_ok=false;
while(std::getline(in,line)){
if(line.size()==0) continue;
if(line[0]=='%') continue;
std::istringstream iss(line);
int M,N,nnz; iss>>M>>N>>nnz;
COO coo; coo.M=M; coo.N=N; coo.rows.reserve(nnz); coo.cols.reserve(nnz); coo.vals.reserve(nnz);
int r,c; double v;
bool one_based=true; // assume 1-based; adjust if we detect 0
std::streampos after_dims = in.tellg();
for(int i=0;i<nnz;++i){
if(!(in>>r>>c>>v)) throw std::runtime_error("Bad MTX data");
if(i==0){ if(r==0 || c==0) one_based=false; }
coo.rows.push_back(r-(one_based?1:0));
coo.cols.push_back(c-(one_based?1:0));
coo.vals.push_back((float)v);
}
return coo;
}
throw std::runtime_error("Invalid MTX header or empty file");
}