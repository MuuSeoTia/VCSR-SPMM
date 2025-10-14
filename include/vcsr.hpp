// vcsr.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <stdexcept>

// VCSR layout tailored for SpMM Case 2 (C stationary, stream columns of A)
// We segment columns of A into width `segw`, then within each segment we bundle rows (size `bundle`),
// sort by per-segment nnz, and pack values/colOffsets in column-major-by-depth for coalesced loads.
struct VCSRSpMM {
  int M{0}, N{0};
  int segw{4};
  int bundle{32};
  // Packed arrays across all groups (in all segments)
  std::vector<int> group_ptr;        // size = num_groups+1 (start index into val/lcol for each group)
  std::vector<int> group_depth;      // max depth per group
  std::vector<int> group_seg_base;   // base column j0 for this group's segment
  std::vector<int> group_rows;       // size = num_groups*bundle (row id per lane)
  std::vector<int> lcol;             // local column offset [0..segw) per packed nnz (padded with -1)
  std::vector<float> val;            // value per packed nnz (undefined if lcol==-1)
};

inline VCSRSpMM csr_to_vcsr_spmm(const CSR &csr, int segw=4, int bundle=32){
  if(bundle<=0) throw std::runtime_error("bundle must be >0");
  VCSRSpMM V; V.M=csr.M; V.N=csr.N; V.segw=segw; V.bundle=bundle;
  const int S = (csr.N + segw - 1)/segw; // number of column segments

  // For each segment, compute per-row nnz and collect (row, list of (local_col, val))
  struct Entry { int lcol; float v; };
  std::vector<std::vector<Entry>> seg_row_entries; // flattened per segment per row
  seg_row_entries.resize((size_t)S * csr.M);

  for(int r=0;r<csr.M;++r){
    for(int p=csr.row_ptr[r]; p<csr.row_ptr[r+1]; ++p){
      int c = csr.col[p]; float v = csr.val[p];
      int sid = c / segw; int l = c % segw;
      seg_row_entries[(size_t)sid*csr.M + r].push_back({l, v});
    }
  }
  // Sort each row's entries within a segment by local col for deterministic packing
  for(int sid=0; sid<S; ++sid){
    for(int r=0;r<csr.M;++r){
      auto &vec = seg_row_entries[(size_t)sid*csr.M + r];
      std::sort(vec.begin(), vec.end(), [](const Entry&a,const Entry&b){return a.lcol<b.lcol;});
    }
  }

  // Build groups per segment: sort rows by nnz desc, then take chunks of `bundle`
  std::vector<int> group_ptr; std::vector<int> group_depth; std::vector<int> group_seg_base; std::vector<int> group_rows;
  std::vector<int> lcol; std::vector<float> val;

  for(int sid=0; sid<S; ++sid){
    // nnz per row in this segment
    std::vector<int> rows(csr.M); std::iota(rows.begin(), rows.end(), 0);
    std::sort(rows.begin(), rows.end(), [&](int a,int b){
      return seg_row_entries[(size_t)sid*csr.M + a].size() > seg_row_entries[(size_t)sid*csr.M + b].size();
    });
    // create bundles
    for(size_t start=0; start<rows.size(); start+=bundle){
      size_t end = std::min(start + (size_t)bundle, rows.size());
      int this_bundle = (int)(end - start);
      if(this_bundle==0) break;
      // determine max depth among rows in this bundle
      int depth=0;
      for(size_t i=start;i<end;++i){
        depth = std::max(depth, (int)seg_row_entries[(size_t)sid*csr.M + rows[i]].size());
      }
      if(depth==0) continue; // skip empty group
      int g_start = (int)val.size();
      group_ptr.push_back(g_start);
      group_depth.push_back(depth);
      group_seg_base.push_back(sid*segw);
      // record rows (pad to bundle with -1 row id => kernel will ignore lanes >= this_bundle)
      for(int lane=0; lane<bundle; ++lane){
        int ridx = (int)start + lane;
        int row = (ridx < (int)rows.size()) ? rows[ridx] : -1;
        group_rows.push_back(row);
      }
      // pack by depth (column-major across rows)
      for(int d=0; d<depth; ++d){
        for(int lane=0; lane<bundle; ++lane){
          int ridx = (int)start + lane;
          int row = (ridx < (int)rows.size()) ? rows[ridx] : -1;
          if(row<0){ lcol.push_back(-1); val.push_back(0.f); continue; }
          auto &vec = seg_row_entries[(size_t)sid*csr.M + row];
          if(d < (int)vec.size()){
            lcol.push_back(vec[d].lcol);
            val.push_back(vec[d].v);
          } else {
            lcol.push_back(-1);
            val.push_back(0.f);
          }
        }
      }
    }
  }
  // close last pointer
  group_ptr.push_back((int)val.size());

  V.group_ptr = std::move(group_ptr);
  V.group_depth = std::move(group_depth);
  V.group_seg_base = std::move(group_seg_base);
  V.group_rows = std::move(group_rows);
  V.lcol = std::move(lcol);
  V.val = std::move(val);
  return V;
}

