// dump_graph_level0.cpp — emit the level-0 adjacency of a saved hnswlib index.
//
// Purpose: measure the C++ graph's actual edge density, to place next to the
// Python reference's measured L0 density (~32.0, at the m0 cap because his
// heuristic() back-fills). hnswlib does NOT back-fill, so its degree should come
// in materially lower — this turns the inferred "sparser C++ build" into a
// measured number and explains the C++/PY merge-cost ratio.
//
// Build (from inside the HNSW-Merger tree, same flags as the harness):
//   g++ -O2 -fopenmp -Wno-write-strings dump_graph_level0.cpp -o dump_graph
//
// Run:
//   ./dump_graph <index.hnsw> <dim> [out.csv]
// Emits, to out.csv (default graph_level0.csv):
//   node_id,degree,neighbours(space-separated)
// and prints degree summary stats (mean/median/max, fraction at cap) to stdout —
// which is exactly what scripts/graph_structure.py consumes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include "hnswlib.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <index.hnsw> <dim> [out.csv]\n", argv[0]);
        return 1;
    }
    std::string index_path = argv[1];
    int dim = atoi(argv[2]);
    std::string out_path = (argc >= 4) ? argv[3] : "graph_level0.csv";

    hnswlib::L2Space space(dim);
    auto *idx = new hnswlib::HierarchicalNSW<float>(&space);
    idx->loadIndex(index_path, &space);   // resizes to the stored max_elements

    size_t n = idx->getCurrentElementCount();
    size_t maxM0 = idx->maxM0_;            // level-0 degree cap (= 2*M by default)
    fprintf(stderr, "loaded %s: %zu nodes, maxM0=%zu\n",
            index_path.c_str(), n, maxM0);

    FILE *f = fopen(out_path.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    fprintf(f, "node_id,degree,neighbours\n");

    std::vector<int> degrees;
    degrees.reserve(n);
    size_t at_cap = 0;
    unsigned long long edge_sum = 0;

    for (hnswlib::tableint i = 0; i < (hnswlib::tableint)n; i++) {
        // level-0 link list: first linklistsizeint is the count, then the ids
        hnswlib::linklistsizeint *ll = idx->get_linklist0(i);
        int deg = (int)idx->getListCount(ll);
        hnswlib::tableint *neigh = (hnswlib::tableint *)(ll + 1);

        degrees.push_back(deg);
        edge_sum += deg;
        if ((size_t)deg >= maxM0) at_cap++;

        fprintf(f, "%u,%d,", (unsigned)i, deg);
        for (int j = 0; j < deg; j++) {
            fprintf(f, "%u%s", (unsigned)neigh[j], (j + 1 < deg) ? " " : "");
        }
        fprintf(f, "\n");
    }
    fclose(f);

    std::sort(degrees.begin(), degrees.end());
    double mean = n ? (double)edge_sum / (double)n : 0.0;
    int median = n ? degrees[n / 2] : 0;
    int dmax = n ? degrees.back() : 0;
    int dmin = n ? degrees.front() : 0;

    printf("nodes=%zu  maxM0=%zu\n", n, maxM0);
    printf("degree: mean=%.4f  median=%d  min=%d  max=%d\n", mean, median, dmin, dmax);
    printf("fraction at cap (deg==maxM0): %.4f  (%zu / %zu)\n",
           n ? (double)at_cap / (double)n : 0.0, at_cap, n);
    printf("wrote %s\n", out_path.c_str());

    // one-line comparison hint
    printf("\n[compare] Python reference L0 density is ~%zu (back-filled to m0). "
           "This graph's mean degree is %.2f — the gap is the build-density "
           "mechanism behind the C++/PY merge-cost ratio.\n", maxM0, mean);

    delete idx;
    return 0;
}
