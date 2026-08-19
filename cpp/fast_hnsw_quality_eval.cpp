#include "fast_hnsw_counting_space.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char *kPrefix = "COURSEPAPER_FASTHNSW_QUERY ";
constexpr const char *kSchema = "coursepaper.fasthnsw.query";
constexpr const char *kInstrumentation = "stock-hnswlib-v0.8.0-counted-l2-v1";
constexpr const char *kHnswlibRevision = "3f3429661187e4c24a490a0f148fc6bc89042b3d";

struct Options {
    std::string index;
    std::string query;
    std::string base;
    std::string index_sha256;
    std::size_t ef = 0;
    std::size_t k = 0;
    std::size_t nq = 0;
    std::size_t dim = 0;
    std::size_t n = 0;
    std::size_t identity_samples = 8;
};

std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::size_t parse_size(const std::string &flag, const std::string &text) {
    if (text.empty() || text[0] == '-') {
        throw std::runtime_error(flag + " requires a non-negative integer");
    }
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(text, &consumed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error(flag + " requires an integer, got: " + text);
    }
    if (consumed != text.size() || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(flag + " integer is out of range: " + text);
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char **argv) {
    std::unordered_map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help") {
            std::cout
                << "usage: fast_hnsw_quality_eval --index INDEX --query QUERY.fvecs"
                << " --base BASE.fvecs --ef EF --k K --nq NQ --dim DIM --n N"
                << " --index-sha256 HEX [--identity-samples N]\n";
            std::exit(0);
        }
        if (flag.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::runtime_error("arguments must be --flag value pairs");
        }
        if (values.count(flag)) {
            throw std::runtime_error("duplicate argument: " + flag);
        }
        values[flag] = argv[++i];
    }

    const std::vector<std::string> allowed = {
        "--index", "--query", "--base", "--ef", "--k", "--nq", "--dim",
        "--n", "--identity-samples", "--index-sha256"};
    for (const auto &entry : values) {
        if (std::find(allowed.begin(), allowed.end(), entry.first) == allowed.end()) {
            throw std::runtime_error("unknown argument: " + entry.first);
        }
    }
    auto required = [&](const std::string &flag) -> const std::string & {
        const auto found = values.find(flag);
        if (found == values.end() || found->second.empty()) {
            throw std::runtime_error("missing required argument: " + flag);
        }
        return found->second;
    };

    Options opt;
    opt.index = required("--index");
    opt.query = required("--query");
    opt.base = required("--base");
    opt.ef = parse_size("--ef", required("--ef"));
    opt.k = parse_size("--k", required("--k"));
    opt.nq = parse_size("--nq", required("--nq"));
    opt.dim = parse_size("--dim", required("--dim"));
    opt.n = parse_size("--n", required("--n"));
    if (values.count("--identity-samples")) {
        opt.identity_samples = parse_size("--identity-samples", values["--identity-samples"]);
    }
    if (values.count("--index-sha256")) {
        opt.index_sha256 = values["--index-sha256"];
    }
    if (opt.index_sha256.size() != 64 ||
        !std::all_of(opt.index_sha256.begin(), opt.index_sha256.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
        throw std::runtime_error("--index-sha256 must be a verified lowercase SHA-256 token");
    }
    if (opt.ef == 0 || opt.k == 0 || opt.nq == 0 || opt.dim == 0 || opt.n == 0) {
        throw std::runtime_error("--ef, --k, --nq, --dim, and --n must be positive");
    }
    if (opt.k > opt.n) {
        throw std::runtime_error("--k cannot exceed --n");
    }
    if (opt.identity_samples == 0) {
        throw std::runtime_error("--identity-samples must be positive");
    }
    return opt;
}

std::uint64_t file_size(std::ifstream &input, const std::string &path) {
    input.seekg(0, std::ios::end);
    const std::streamoff end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine file size: " + path);
    }
    input.seekg(0, std::ios::beg);
    return static_cast<std::uint64_t>(end);
}

std::uint64_t checked_fvec_file_size(std::size_t rows, std::size_t dim) {
    const std::uint64_t record = sizeof(std::int32_t) +
        static_cast<std::uint64_t>(dim) * sizeof(float);
    if (rows > std::numeric_limits<std::uint64_t>::max() / record) {
        throw std::runtime_error("fvecs expected size overflow");
    }
    return static_cast<std::uint64_t>(rows) * record;
}

std::vector<float> read_queries(const std::string &path, std::size_t nq, std::size_t dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open query fvecs: " + path);
    }
    const std::uint64_t actual_size = file_size(input, path);
    const std::uint64_t expected_size = checked_fvec_file_size(nq, dim);
    if (actual_size != expected_size) {
        throw std::runtime_error("query fvecs size does not match --nq/--dim");
    }
    if (nq > std::numeric_limits<std::size_t>::max() / dim) {
        throw std::runtime_error("query allocation size overflow");
    }
    std::vector<float> data(nq * dim);
    for (std::size_t row = 0; row < nq; ++row) {
        std::int32_t row_dim = 0;
        input.read(reinterpret_cast<char *>(&row_dim), sizeof(row_dim));
        if (!input || row_dim != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("query fvecs dimension mismatch at row " + std::to_string(row));
        }
        input.read(reinterpret_cast<char *>(data.data() + row * dim),
                   static_cast<std::streamsize>(dim * sizeof(float)));
        if (!input) {
            throw std::runtime_error("truncated query fvecs at row " + std::to_string(row));
        }
    }
    return data;
}

void validate_platform() {
    static_assert(sizeof(float) == 4, "FastKCNA stock serialization requires float32");
    static_assert(sizeof(unsigned int) == 4, "FastKCNA internal IDs require uint32");
    static_assert(sizeof(std::size_t) == 8, "FastKCNA file was produced with 64-bit size_t");
    const std::uint16_t marker = 1;
    if (*reinterpret_cast<const unsigned char *>(&marker) != 1) {
        throw std::runtime_error("FastKCNA stock serialization requires little-endian host");
    }
}

void validate_index_layout(const hnswlib::HierarchicalNSW<float> &index,
                           std::size_t n, std::size_t dim) {
    if (index.cur_element_count.load() != n) {
        throw std::runtime_error("index element count does not match --n");
    }
    if (index.max_elements_ < n) {
        throw std::runtime_error("index max element count is smaller than --n");
    }
    if (index.offsetLevel0_ != 0 || index.offsetData_ != index.size_links_level0_) {
        throw std::runtime_error("index level-0/data offsets are not stock hnswlib layout");
    }
    const std::size_t vector_bytes = dim * sizeof(float);
    if (index.label_offset_ < index.offsetData_ ||
        index.label_offset_ - index.offsetData_ != vector_bytes) {
        throw std::runtime_error("index stored-vector dimension/type does not match --dim float32");
    }
    if (index.size_data_per_element_ != index.label_offset_ + sizeof(hnswlib::labeltype)) {
        throw std::runtime_error("index label/element layout is not stock 64-bit hnswlib layout");
    }
}

void validate_label_permutation(const hnswlib::HierarchicalNSW<float> &index, std::size_t n) {
    std::vector<unsigned char> seen(n, 0);
    for (std::size_t internal = 0; internal < n; ++internal) {
        const hnswlib::labeltype label =
            index.getExternalLabel(static_cast<hnswlib::tableint>(internal));
        if (label >= n) {
            throw std::runtime_error("index external label outside [0,n)");
        }
        if (seen[label]) {
            throw std::runtime_error("index external labels are not unique");
        }
        seen[label] = 1;
    }
}

std::vector<std::size_t> sample_labels(std::size_t n, std::size_t requested) {
    const std::size_t count = std::min(n, requested);
    std::vector<std::size_t> labels;
    labels.reserve(count);
    if (count == 1) {
        labels.push_back(0);
        return labels;
    }
    for (std::size_t i = 0; i < count; ++i) {
        // Evenly-spaced deterministic sample, including first and last rows.
        labels.push_back((i * (n - 1)) / (count - 1));
    }
    return labels;
}

std::size_t validate_vector_samples(hnswlib::HierarchicalNSW<float> &index,
                                    const std::string &base_path,
                                    std::size_t n, std::size_t dim,
                                    std::size_t requested) {
    std::ifstream input(base_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open base fvecs: " + base_path);
    }
    const std::uint64_t actual_size = file_size(input, base_path);
    const std::uint64_t expected_size = checked_fvec_file_size(n, dim);
    if (actual_size != expected_size) {
        throw std::runtime_error("base fvecs size does not match --n/--dim");
    }
    const std::uint64_t record_bytes = sizeof(std::int32_t) +
        static_cast<std::uint64_t>(dim) * sizeof(float);
    std::vector<float> source(dim);
    const std::vector<std::size_t> labels = sample_labels(n, requested);
    for (const std::size_t label : labels) {
        input.clear();
        input.seekg(static_cast<std::streamoff>(label * record_bytes), std::ios::beg);
        std::int32_t source_dim = 0;
        input.read(reinterpret_cast<char *>(&source_dim), sizeof(source_dim));
        input.read(reinterpret_cast<char *>(source.data()),
                   static_cast<std::streamsize>(dim * sizeof(float)));
        if (!input || source_dim != static_cast<std::int32_t>(dim)) {
            throw std::runtime_error("base fvecs dimension/truncation at sampled row " +
                                     std::to_string(label));
        }
        const std::vector<float> stored = index.getDataByLabel<float>(label);
        if (stored.size() != dim ||
            std::memcmp(stored.data(), source.data(), dim * sizeof(float)) != 0) {
            throw std::runtime_error("stored index vector differs from base row/label " +
                                     std::to_string(label));
        }
    }
    return labels.size();
}

std::vector<hnswlib::labeltype> nearest_first(
        std::priority_queue<std::pair<float, hnswlib::labeltype>> queue,
        std::size_t expected_k) {
    if (queue.size() != expected_k) {
        throw std::runtime_error("stock hnswlib returned fewer than k labels");
    }
    std::vector<hnswlib::labeltype> labels;
    labels.reserve(queue.size());
    while (!queue.empty()) {
        labels.push_back(queue.top().second);  // stock heap pops farthest first
        queue.pop();
    }
    std::reverse(labels.begin(), labels.end());
    return labels;
}

void emit_record(const Options &opt, std::uint64_t total, std::size_t checked,
                 const std::vector<std::vector<hnswlib::labeltype>> &results) {
    const double mean = static_cast<double>(total) / static_cast<double>(opt.nq);
    std::ostringstream out;
    out << kPrefix << "{"
        << "\"schema\":\"" << kSchema << "\","
        << "\"version\":1,"
        << "\"instrumentation\":\"" << kInstrumentation << "\","
        << "\"hnswlib_revision\":\"" << kHnswlibRevision << "\","
        << "\"index_sha256\":\"" << json_escape(opt.index_sha256) << "\","
        << "\"metric\":\"squared_l2_float32\","
        << "\"query_threading\":\"single\","
        << "\"k\":" << opt.k << ","
        << "\"ef_search\":" << opt.ef << ","
        << "\"query_count\":" << opt.nq << ","
        << "\"distance_evaluations_total\":" << total << ","
        << "\"distance_evaluations_mean\":" << std::setprecision(17) << mean << ","
        << "\"index_element_count\":" << opt.n << ","
        << "\"dim\":" << opt.dim << ","
        << "\"index_label_permutation_validated\":true,"
        << "\"identity_samples_checked\":" << checked << ","
        << "\"result_labels\":[";
    for (std::size_t qi = 0; qi < results.size(); ++qi) {
        if (qi) out << ',';
        out << '[';
        for (std::size_t ri = 0; ri < results[qi].size(); ++ri) {
            if (ri) out << ',';
            out << results[qi][ri];
        }
        out << ']';
    }
    out << "]}";
    std::cout << out.str() << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    try {
        validate_platform();
        const Options opt = parse_options(argc, argv);
        const std::vector<float> queries = read_queries(opt.query, opt.nq, opt.dim);

        CountingL2Space space(opt.dim);
        hnswlib::HierarchicalNSW<float> index(&space, opt.index);
        validate_index_layout(index, opt.n, opt.dim);
        validate_label_permutation(index, opt.n);
        const std::size_t checked = validate_vector_samples(
            index, opt.base, opt.n, opt.dim, opt.identity_samples);

        index.setEf(opt.ef);
        space.reset();
        std::vector<std::vector<hnswlib::labeltype>> results;
        results.reserve(opt.nq);
        for (std::size_t qi = 0; qi < opt.nq; ++qi) {
            results.push_back(nearest_first(
                index.searchKnn(queries.data() + qi * opt.dim, opt.k), opt.k));
        }
        const std::uint64_t total = space.completed();
        emit_record(opt, total, checked, results);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fast_hnsw_quality_eval: " << error.what() << '\n';
        return 1;
    }
}
