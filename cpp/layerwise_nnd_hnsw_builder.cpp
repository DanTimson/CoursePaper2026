#include "kgraph.h"
#include "kgraph-data.h"
#include "graph_utils.h"

#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char *kPrefix = "COURSEPAPER_LAYERWISE_NND ";
constexpr const char *kSchema = "coursepaper.layerwise_nnd_hnsw.build";
constexpr const char *kInstrumentation = "fastkcna-metric-boundary-layerwise-v2";

struct Options {
    std::string data;
    std::string output;
    std::string levels_file;
    unsigned K = 500, L = 500, S = 12, R = 100, iterations = 6;
    unsigned seed = 2024, threads = 1, controls = 100, M = 16;
    float delta = 0.002f, recall = 0.98f;
};

unsigned parse_unsigned(const std::string &flag, const std::string &text) {
    if (text.empty() || text[0] == '-') throw std::runtime_error(flag + " requires a nonnegative integer");
    std::size_t used = 0;
    unsigned long value = std::stoul(text, &used, 10);
    if (used != text.size() || value > std::numeric_limits<unsigned>::max())
        throw std::runtime_error(flag + " is out of range: " + text);
    return static_cast<unsigned>(value);
}

float parse_float(const std::string &flag, const std::string &text) {
    std::size_t used = 0;
    float value = std::stof(text, &used);
    if (used != text.size()) throw std::runtime_error(flag + " requires a float: " + text);
    return value;
}

Options parse_options(int argc, char **argv) {
    std::unordered_map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        std::string flag(argv[i]);
        if (flag == "--help") {
            std::cout << "usage: layerwise_nnd_hnsw_builder --data DATA.lshkit --output INDEX.hnsw "
                         "[--K 500 --L 500 --S 12 --R 100 --iter 6 --seed 2024 --delta 0.002 "
                         "--controls 100 --recall 0.98 --M 16 --threads 1] "
                         "[--levels-file validation-levels.txt]\n";
            std::exit(0);
        }
        if (flag.rfind("--", 0) != 0 || i + 1 >= argc)
            throw std::runtime_error("arguments must be --flag value pairs");
        if (values.count(flag)) throw std::runtime_error("duplicate argument: " + flag);
        values[flag] = argv[++i];
    }
    const std::set<std::string> allowed = {"--data", "--output", "--levels-file", "--K", "--L", "--S", "--R", "--iter", "--seed", "--delta", "--controls", "--recall", "--M", "--threads"};
    for (const auto &v : values) if (!allowed.count(v.first)) throw std::runtime_error("unknown argument: " + v.first);
    auto required = [&](const std::string &name) -> std::string {
        auto it = values.find(name);
        if (it == values.end() || it->second.empty()) throw std::runtime_error("missing required argument: " + name);
        return it->second;
    };
    Options o;
    o.data = required("--data"); o.output = required("--output");
    if (values.count("--levels-file")) o.levels_file = values["--levels-file"];
    auto u = [&](const char *name, unsigned &field) { if (values.count(name)) field = parse_unsigned(name, values[name]); };
    u("--K",o.K); u("--L",o.L); u("--S",o.S); u("--R",o.R); u("--iter",o.iterations);
    u("--seed",o.seed); u("--controls",o.controls); u("--M",o.M); u("--threads",o.threads);
    if (values.count("--delta")) o.delta = parse_float("--delta", values["--delta"]);
    if (values.count("--recall")) o.recall = parse_float("--recall", values["--recall"]);
    if (!o.K || !o.L || !o.S || !o.R || !o.iterations || !o.controls || !o.M || !o.threads)
        throw std::runtime_error("integer construction parameters must be positive");
    if (!(o.delta > 0.0f) || !(o.recall > 0.0f && o.recall <= 1.0f))
        throw std::runtime_error("delta/recall are outside supported ranges");
    if (o.M != 16) throw std::runtime_error("canonical baseline fixes M=16");
    return o;
}

std::vector<int> read_levels(const std::string &path, unsigned n) {
    std::ifstream in(path.c_str());
    if (!in) throw std::runtime_error("cannot open validation levels file: " + path);
    std::vector<int> levels; long long value;
    while (in >> value) {
        if (value < 0 || value > std::numeric_limits<int>::max())
            throw std::runtime_error("validation level is outside nonnegative int range");
        levels.push_back(static_cast<int>(value));
    }
    if (!in.eof()) throw std::runtime_error("malformed validation levels file: " + path);
    if (levels.size() != n) throw std::runtime_error("validation levels count does not match dataset");
    return levels;
}

template <typename Oracle>
std::vector<unsigned> diversify(unsigned query, const std::vector<unsigned> &raw,
                                unsigned limit, unsigned occupancy, const Oracle &oracle) {
    std::vector<unsigned> ids;
    ids.reserve(raw.size());
    for (unsigned id : raw) if (id < occupancy && id != query) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    // Stock getNeighborsByHeuristic2 returns immediately (and performs no
    // candidate-to-selected metric work) when fewer than limit candidates exist.
    if (ids.size() < limit) return ids;
    std::vector<std::pair<float, unsigned>> ranked;
    ranked.reserve(ids.size());
    for (unsigned id : ids) ranked.push_back(std::make_pair(oracle(query, id), id));
    // Pinned stock first moves (-distance,id) pairs into the default
    // priority_queue<pair<...>>. Its lexicographic top is nearest distance and,
    // at exactly equal distance, the greatest internal ID.
    std::sort(ranked.begin(), ranked.end(), [](const std::pair<float,unsigned> &a, const std::pair<float,unsigned> &b) {
        return a.first < b.first || (a.first == b.first && a.second > b.second);
    });
    std::vector<unsigned> selected;
    selected.reserve(std::min<std::size_t>(limit, ranked.size()));
    for (const auto &candidate : ranked) {
        if (selected.size() >= limit) break;
        bool good = true;
        for (unsigned chosen : selected) {
            // Stock hnswlib getNeighborsByHeuristic2 rejects only when the
            // candidate-to-selected distance is strictly smaller.
            if (oracle(candidate.second, chosen) < candidate.first) { good = false; break; }
        }
        if (good) selected.push_back(candidate.second);
    }
    return selected;
}

template <typename Oracle>
void convert_layer(const std::vector<std::vector<unsigned>> &candidates,
                   unsigned occupancy, unsigned initial_limit, unsigned final_capacity,
                   const Oracle &oracle, std::vector<std::vector<unsigned>> &output,
                   unsigned &initial_max_degree, std::vector<unsigned> &initial_node0_neighbors,
                   unsigned &final_max_degree) {
    std::vector<std::set<unsigned>> adjacency(occupancy);
    initial_max_degree = 0;
    initial_node0_neighbors.clear();
    oracle.distance_accounting_set_phase(kgraph::DISTANCE_PHASE_NEIGHBOR_PRUNE);
    for (unsigned i = 0; i < occupancy; ++i) {
        std::vector<unsigned> chosen = diversify(i, candidates[i], initial_limit, occupancy, oracle);
        initial_max_degree = std::max(initial_max_degree, static_cast<unsigned>(chosen.size()));
        if (i == 0) initial_node0_neighbors = chosen;
        for (unsigned j : chosen) adjacency[i].insert(j);
    }
    // One reciprocity insertion: form the undirected union of selected arcs.
    for (unsigned i = 0; i < occupancy; ++i) {
        std::vector<unsigned> current(adjacency[i].begin(), adjacency[i].end());
        for (unsigned j : current) adjacency[j].insert(i);
    }
    // Overflow processing only removes undirected edges, so it cannot create a
    // new overflow or break reciprocity. Ascending node order is deterministic.
    oracle.distance_accounting_set_phase(kgraph::DISTANCE_PHASE_REVERSE_REPAIR);
    for (unsigned i = 0; i < occupancy; ++i) {
        if (adjacency[i].size() <= final_capacity) continue;
        std::vector<unsigned> current(adjacency[i].begin(), adjacency[i].end());
        std::vector<unsigned> kept_vec = diversify(i, current, final_capacity, occupancy, oracle);
        std::set<unsigned> kept(kept_vec.begin(), kept_vec.end());
        for (unsigned j : current) {
            if (!kept.count(j)) { adjacency[i].erase(j); adjacency[j].erase(i); }
        }
    }
    output.assign(occupancy, std::vector<unsigned>());
    final_max_degree = 0;
    for (unsigned i = 0; i < occupancy; ++i) {
        output[i].assign(adjacency[i].begin(), adjacency[i].end());
        final_max_degree = std::max(final_max_degree, static_cast<unsigned>(output[i].size()));
    }
}

void validate_graphs(const std::vector<std::vector<std::vector<unsigned>>> &graphs,
                     const std::vector<unsigned> &occupancies, const std::vector<int> &levels,
                     const std::vector<unsigned> &labels) {
    for (unsigned layer = 0; layer < occupancies.size(); ++layer) {
        unsigned cap = layer == 0 ? 32 : 16;
        unsigned occ = occupancies[layer];
        for (unsigned i = 0; i < occ; ++i) {
            if (levels[labels[i]] < static_cast<int>(layer)) throw std::runtime_error("serialized layer membership mismatch");
            if (graphs[layer][i].size() > cap) throw std::runtime_error("degree cap exceeded");
            std::set<unsigned> unique;
            for (unsigned j : graphs[layer][i]) {
                if (j >= occ || j == i || !unique.insert(j).second) throw std::runtime_error("invalid layer edge");
                if (!std::binary_search(graphs[layer][j].begin(), graphs[layer][j].end(), i))
                    throw std::runtime_error("non-reciprocal layer edge");
            }
        }
    }
}

void print_unsigned_array(std::ostream &out, const std::vector<unsigned> &values) {
    out << '['; for (std::size_t i=0;i<values.size();++i) { if(i) out << ','; out << values[i]; } out << ']';
}
void print_u64_array(std::ostream &out, const std::vector<uint64_t> &values) {
    out << '['; for (std::size_t i=0;i<values.size();++i) { if(i) out << ','; out << values[i]; } out << ']';
}
void print_unsigned_matrix(std::ostream &out, const std::vector<std::vector<unsigned>> &values) {
    out << '[';
    for (std::size_t i=0;i<values.size();++i) { if(i) out << ','; print_unsigned_array(out, values[i]); }
    out << ']';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options opt = parse_options(argc, argv);
        omp_set_num_threads(static_cast<int>(opt.threads));
        const unsigned n = kgraph::getPointNum(opt.data.c_str());
        if (n == 0) throw std::runtime_error("empty/invalid lshkit dataset");

        std::vector<int> levels;
        if (opt.levels_file.empty()) kgraph::getLevel(n, levels, opt.M, opt.seed);
        else levels = read_levels(opt.levels_file, n);
        const int max_level = *std::max_element(levels.begin(), levels.end());
        std::vector<unsigned> labels, file2data;
        std::vector<int> occupancy_int;
        kgraph::getMapping(n, levels, labels, file2data, occupancy_int);
        std::vector<unsigned> occupancies(occupancy_int.begin(), occupancy_int.end());
        if (occupancies.empty() || occupancies[0] != n) throw std::runtime_error("invalid hierarchy occupancy");

        kgraph::Matrix<float> matrix;
        matrix.load_lshkit(opt.data, file2data.data());
        kgraph::MatrixOracle<float, kgraph::metric::l2sqr> oracle(matrix);
        oracle.configure_distance_accounting(static_cast<unsigned>(max_level + 1), opt.threads);

        std::vector<std::vector<std::vector<unsigned>>> graphs(max_level + 1);
        for (int layer = 0; layer <= max_level; ++layer)
            graphs[layer].resize(occupancies[layer]);
        std::vector<unsigned> invocations(max_level + 1, 0), effective_K(max_level + 1, 0),
            effective_L(max_level + 1, 0), effective_S(max_level + 1, 0),
            effective_iterations(max_level + 1, 0), actual_iterations(max_level + 1, 0),
            effective_controls(max_level + 1, 0);
        std::vector<uint64_t> diagnostic_n_comps(max_level + 1, 0);
        std::vector<unsigned> initial_max_degrees(max_level + 1, 0), final_max_degrees(max_level + 1, 0);
        std::vector<std::vector<unsigned>> initial_node0_neighbors(max_level + 1);

        kgraph::KGraph::IndexParams params;
        params.pg_type = kgraph::INDEX_KNNG;
        params.K=opt.K; params.L=opt.L; params.S=opt.S; params.R=opt.R;
        params.iterations=opt.iterations; params.seed=opt.seed; params.delta=opt.delta;
        params.controls=opt.controls; params.recall=opt.recall;
        // pg0 does not consume FastHNSW search/bridge fields. Values are only
        // made safe for upstream's diagnostic control-size calculation.
        params.search_L=opt.L; params.search_K=opt.K; params.nthreads=opt.threads;
        params.nsg_R=opt.M; params.loop_i=0; params.step=10; params.alpha=60; params.tau=0;

        for (int layer = max_level; layer >= 0; --layer) {
            const unsigned occ = occupancies[layer];
            const unsigned initial_limit = opt.M;
            const unsigned final_capacity = layer == 0 ? 2 * opt.M : opt.M;
            oracle.distance_accounting_disable();
            oracle.set_size(occ);
            oracle.set_distance_accounting_layer(static_cast<unsigned>(layer));
            if (occ < 2) {
                effective_K[layer]=effective_L[layer]=effective_S[layer]=effective_iterations[layer]=0;
                effective_controls[layer]=0;
                continue;
            }
            effective_K[layer] = std::min(opt.K, occ - 1);
            effective_L[layer] = std::min(opt.L, occ - 1);
            effective_S[layer] = occ <= opt.K ? occ - 1 : std::min(opt.S, occ - 1);
            effective_iterations[layer] = occ <= opt.K ? 0 : opt.iterations;
            effective_controls[layer] = std::min(opt.controls, occ - 1);
            std::vector<std::vector<unsigned>> candidates;
            unsigned ignored_entry = 0;
            kgraph::KGraph::IndexInfo info{};
            std::unique_ptr<kgraph::KGraph> index(kgraph::KGraph::create());
            // Pinned KGraph uses both params.seed-driven mt19937 instances and
            // legacy std::random_shuffle/std::rand. Reset the latter before
            // every invocation so one layer cannot inherit RNG state from a
            // preceding layer: every candidate build really starts afresh.
            std::srand(opt.seed);
            ++invocations[layer];
            index->build(oracle, params, &info, candidates, ignored_entry);
            actual_iterations[layer] = info.iterations;
            diagnostic_n_comps[layer] = info.n_comps;
            if (candidates.size() != occ) throw std::runtime_error("candidate builder returned wrong layer size");
            std::vector<std::vector<unsigned>> converted;
            convert_layer(candidates, occ, initial_limit, final_capacity, oracle, converted,
                          initial_max_degrees[layer], initial_node0_neighbors[layer], final_max_degrees[layer]);
            for (unsigned i=0;i<occ;++i) graphs[layer][i].swap(converted[i]);
        }
        oracle.distance_accounting_disable();
        validate_graphs(graphs, occupancies, levels, labels);

        // getMapping orders equal-level labels ascending; internal 0 is therefore
        // the smallest original label at the maximum level.
        const unsigned entry_point = 0;
        const unsigned entry_label = labels[entry_point];
        kgraph::KGraph::IndexParams serialization_params = params;
        serialization_params.nsg_R = opt.M;
        serialization_params.search_K = opt.K;
        kgraph::save_hnsw(matrix, graphs, serialization_params, entry_point, levels, labels, opt.output.c_str());
        std::ifstream check(opt.output.c_str(), std::ios::binary | std::ios::ate);
        if (!check || check.tellg() <= 0) throw std::runtime_error("stock HNSW serialization failed");

        const kgraph::DistanceAccountingSnapshot counts = oracle.distance_accounting_snapshot();
        uint64_t layer_sum=0, phase_sum=0;
        for (uint64_t v: counts.layer_totals) layer_sum += v;
        for (uint64_t v: counts.phase_totals) phase_sum += v;
        if (layer_sum != counts.total || phase_sum != counts.total) throw std::runtime_error("distance-accounting additive invariant failed");
        if (counts.phase_totals[kgraph::DISTANCE_PHASE_CONSTRUCTION_SEARCH] != 0)
            throw std::runtime_error("forbidden construction_search work was observed");

        std::ostringstream out;
        out << kPrefix << '{'
            << "\"schema\":\"" << kSchema << "\",\"version\":2,"
            << "\"instrumentation\":\"" << kInstrumentation << "\","
            << "\"metric\":\"squared_l2_float32\",\"construction_threads\":" << opt.threads << ','
            << "\"n\":" << n << ",\"dim\":" << matrix.dim() << ",\"M\":" << opt.M
            << ",\"initial_diversification_limit\":" << opt.M
            << ",\"base_degree_cap\":" << 2*opt.M << ",\"upper_degree_cap\":" << opt.M
            << ",\"diversification_tie_rule\":\"stock-nearest-distance-then-descending-internal-id\""
            << ",\"level_seed\":" << opt.seed << ",\"max_level\":" << max_level
            << ",\"entry_point_internal\":" << entry_point << ",\"entry_point_label\":" << entry_label
            << ",\"level_rule\":\"" << (opt.levels_file.empty() ? "fastkcna-getLevel-stock-hnswlib-equivalent" : "validation-injected-level-vector") << "\""
            << ",\"candidate_rng_rule\":\"params-seed-mt19937-plus-per-invocation-srand-seed\""
            << ",\"layer_occupancies\":"; print_unsigned_array(out,occupancies);
        out << ",\"candidate_build_invocations\":"; print_unsigned_array(out,invocations);
        out << ",\"effective_K\":"; print_unsigned_array(out,effective_K);
        out << ",\"effective_L\":"; print_unsigned_array(out,effective_L);
        out << ",\"effective_S\":"; print_unsigned_array(out,effective_S);
        out << ",\"effective_iterations\":"; print_unsigned_array(out,effective_iterations);
        out << ",\"actual_iterations\":"; print_unsigned_array(out,actual_iterations);
        out << ",\"effective_controls\":"; print_unsigned_array(out,effective_controls);
        out << ",\"diagnostic_upstream_n_comps\":"; print_u64_array(out,diagnostic_n_comps);
        out << ",\"initial_selected_max_degree\":"; print_unsigned_array(out,initial_max_degrees);
        out << ",\"final_max_degree\":"; print_unsigned_array(out,final_max_degrees);
        out << ",\"initial_node0_selected_neighbors\":"; print_unsigned_matrix(out,initial_node0_neighbors);
        out << ",\"phase_totals\":{"
            << "\"knng_candidate\":" << counts.phase_totals[kgraph::DISTANCE_PHASE_KNNG_CANDIDATE] << ','
            << "\"construction_search\":" << counts.phase_totals[kgraph::DISTANCE_PHASE_CONSTRUCTION_SEARCH] << ','
            << "\"neighbor_prune\":" << counts.phase_totals[kgraph::DISTANCE_PHASE_NEIGHBOR_PRUNE] << ','
            << "\"reverse_repair\":" << counts.phase_totals[kgraph::DISTANCE_PHASE_REVERSE_REPAIR] << ','
            << "\"other_construction\":" << counts.phase_totals[kgraph::DISTANCE_PHASE_OTHER_CONSTRUCTION] << "}"
            << ",\"layer_totals\":"; print_u64_array(out,counts.layer_totals);
        out << ",\"construction_total\":" << counts.total
            << ",\"candidate_parameters\":{\"K\":" << opt.K << ",\"L\":" << opt.L
            << ",\"S\":" << opt.S << ",\"R\":" << opt.R << ",\"iter\":" << opt.iterations
            << ",\"seed\":" << opt.seed << ",\"delta\":" << std::setprecision(9) << opt.delta
            << ",\"controls\":" << opt.controls << ",\"recall_stop\":" << opt.recall << "}"
            << ",\"structural_validation\":{\"membership\":true,\"no_self_or_duplicate\":true,\"degree_caps\":true,\"reciprocal\":true}"
            << '}';
        std::cout << out.str() << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "layerwise_nnd_hnsw_builder: " << e.what() << std::endl;
        return 2;
    }
}
