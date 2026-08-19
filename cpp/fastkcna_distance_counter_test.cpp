// Low-level exact validation for the external FastKCNA counting oracle.
// Compile/run commands are documented in FASTKCNA_DISTANCE_ACCOUNTING.md.
#include "kgraph.h"
#include "kgraph-data.h"
#include <cstdint>
#include <iostream>
#include <omp.h>

int main()
{
    kgraph::Matrix<float> matrix(4, 8);
    matrix.zero();
    for (unsigned i = 0; i < matrix.size(); ++i)
        for (unsigned j = 0; j < matrix.dim(); ++j)
            matrix[i][j] = static_cast<float>(i * 10 + j);

    kgraph::MatrixOracle<float, kgraph::metric::l2sqr> oracle(matrix);
    oracle.configure_distance_accounting(2, 4);
    for (unsigned i = 0; i < 5; ++i)
        (void)oracle(i % 4, (i + 1) % 4); // disabled diagnostic scope

    oracle.set_distance_accounting_layer(0);
    oracle.distance_accounting_set_phase(kgraph::DISTANCE_PHASE_KNNG_CANDIDATE);
    for (unsigned i = 0; i < 7; ++i)
        (void)oracle(i % 4, (i + 1) % 4);

    oracle.set_distance_accounting_layer(1);
    oracle.distance_accounting_set_phase(kgraph::DISTANCE_PHASE_CONSTRUCTION_SEARCH);
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < 1000; ++i)
        (void)oracle(static_cast<unsigned>(i) % 4, (static_cast<unsigned>(i) + 1) % 4);
    oracle.distance_accounting_disable();

    kgraph::DistanceAccountingSnapshot counts = oracle.distance_accounting_snapshot();
    bool ok = counts.total == 1007 &&
              counts.phase_totals[kgraph::DISTANCE_PHASE_KNNG_CANDIDATE] == 7 &&
              counts.phase_totals[kgraph::DISTANCE_PHASE_CONSTRUCTION_SEARCH] == 1000 &&
              counts.phase_totals[kgraph::DISTANCE_PHASE_NEIGHBOR_PRUNE] == 0 &&
              counts.phase_totals[kgraph::DISTANCE_PHASE_REVERSE_REPAIR] == 0 &&
              counts.phase_totals[kgraph::DISTANCE_PHASE_OTHER_CONSTRUCTION] == 0 &&
              counts.layer_totals.size() == 2 &&
              counts.layer_totals[0] == 7 && counts.layer_totals[1] == 1000;
    std::cout << "DIRECT_DISTANCE_COUNTER total=" << counts.total
              << " layer0=" << counts.layer_totals[0]
              << " layer1=" << counts.layer_totals[1]
              << " excluded_disabled=5 status=" << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
