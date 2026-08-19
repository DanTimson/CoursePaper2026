#include "vendor/hnswlib/hnswlib.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Options { std::string index, base, levels; std::size_t n=0, dim=0; };
std::size_t number(const std::string &flag, const std::string &text) {
    std::size_t used=0; unsigned long long value=std::stoull(text,&used);
    if (used!=text.size() || value==0 || value>std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("invalid "+flag);
    return static_cast<std::size_t>(value);
}
Options parse(int argc,char **argv) {
    std::unordered_map<std::string,std::string> v;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("arguments must be pairs");
        v[argv[i]] = argv[i + 1];
    }
    for(const auto &x:v) if(x.first!="--index"&&x.first!="--base"&&x.first!="--levels"&&x.first!="--n"&&x.first!="--dim") throw std::runtime_error("unknown argument");
    auto req=[&](const char *x){if(!v.count(x))throw std::runtime_error(std::string("missing ")+x);return v[x];};
    Options o; o.index=req("--index");o.base=req("--base");o.levels=req("--levels");o.n=number("--n",req("--n"));o.dim=number("--dim",req("--dim"));return o;
}
std::vector<int> read_levels(const std::string &path,std::size_t n){
    std::ifstream in(path); if(!in)throw std::runtime_error("cannot open levels");
    std::vector<int> out; long long x; while(in>>x){if(x<0||x>std::numeric_limits<int>::max())throw std::runtime_error("invalid level");out.push_back(static_cast<int>(x));}
    if (!in.eof() || out.size() != n) throw std::runtime_error("levels shape mismatch");
    return out;
}
std::vector<float> read_base(const std::string &path,std::size_t n,std::size_t dim){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open base");
    std::vector<float> out(n*dim);for(std::size_t i=0;i<n;++i){std::int32_t d=0;in.read(reinterpret_cast<char*>(&d),4);if(!in||d!=static_cast<std::int32_t>(dim))throw std::runtime_error("base row mismatch");in.read(reinterpret_cast<char*>(out.data()+i*dim),dim*4);if(!in)throw std::runtime_error("truncated base");}
    char extra;if(in.read(&extra,1))throw std::runtime_error("extra base bytes");return out;
}
}
int main(int argc,char **argv){
 try{
    Options o=parse(argc,argv);auto levels=read_levels(o.levels,o.n);auto base=read_base(o.base,o.n,o.dim);
    hnswlib::L2Space space(o.dim);hnswlib::HierarchicalNSW<float> idx(&space,o.index);
    if(idx.cur_element_count.load()!=o.n||idx.M_!=16||idx.maxM_!=16||idx.maxM0_!=32)throw std::runtime_error("header/cap mismatch");
    int maxlevel=*std::max_element(levels.begin(),levels.end());if(idx.maxlevel_!=maxlevel)throw std::runtime_error("max level mismatch");
    std::size_t min_top=o.n;
    for(std::size_t label=0;label<o.n;++label)if(levels[label]==maxlevel)min_top=std::min(min_top,label);
    if(idx.getExternalLabel(idx.enterpoint_node_)!=min_top)throw std::runtime_error("entry point mismatch");
    for(std::size_t i=0;i<o.n;++i){
      std::size_t label=idx.getExternalLabel(static_cast<hnswlib::tableint>(i));if(label>=o.n||idx.element_levels_[i]!=levels[label])throw std::runtime_error("label/level mismatch");
      auto stored=idx.getDataByLabel<float>(label);if(stored.size()!=o.dim||std::memcmp(stored.data(),base.data()+label*o.dim,o.dim*4))throw std::runtime_error("stored vector mismatch");
    }
    std::uint64_t directed=0;
    for(int layer=0;layer<=maxlevel;++layer){std::size_t cap=layer?16:32;
      for(std::size_t i=0;i<o.n;++i){if(levels[idx.getExternalLabel(static_cast<hnswlib::tableint>(i))]<layer)continue;
        auto *list=idx.get_linklist_at_level(static_cast<hnswlib::tableint>(i),layer);std::size_t degree=idx.getListCount(list);if(degree>cap)throw std::runtime_error("degree cap exceeded");
        auto *neighbors=reinterpret_cast<hnswlib::tableint*>(list+1);std::set<hnswlib::tableint> unique;
        for(std::size_t k=0;k<degree;++k){auto j=neighbors[k];if(j>=o.n||j==i||!unique.insert(j).second)throw std::runtime_error("invalid edge");
          if (levels[idx.getExternalLabel(j)] < layer) throw std::runtime_error("cross-layer edge");
          auto *back = idx.get_linklist_at_level(j, layer);
          auto *bn = reinterpret_cast<hnswlib::tableint *>(back + 1);
          std::size_t bd = idx.getListCount(back);
          if (std::find(bn, bn + bd, i) == bn + bd) throw std::runtime_error("nonreciprocal edge");
          ++directed;
        }
      }
    }
    std::cout<<"COURSEPAPER_LAYERWISE_VALIDATE {\"stock_load\":true,\"labels_vectors_levels\":true,\"degree_caps\":true,\"reciprocal\":true,\"entry_point_label\":"<<min_top<<",\"directed_edges\":"<<directed<<"}\n";return 0;
 }catch(const std::exception &e){std::cerr<<"layerwise_nnd_hnsw_validate: "<<e.what()<<"\n";return 2;}
}
