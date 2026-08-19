#include "fast_hnsw_counting_space.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

std::set<hnswlib::labeltype> labels_of(
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result) {
    std::set<hnswlib::labeltype> labels;
    while (!result.empty()) {
        labels.insert(result.top().second);
        result.pop();
    }
    return labels;
}

std::set<hnswlib::labeltype> brute_labels(const std::vector<std::vector<float>> &points,
                                           const float *query, std::size_t k) {
    std::vector<std::pair<float, hnswlib::labeltype>> distances;
    for (std::size_t label = 0; label < points.size(); ++label) {
        float distance = 0.0f;
        for (std::size_t d = 0; d < points[label].size(); ++d) {
            const float delta = query[d] - points[label][d];
            distance += delta * delta;
        }
        distances.emplace_back(distance, label);
    }
    std::sort(distances.begin(), distances.end());
    std::set<hnswlib::labeltype> labels;
    for (std::size_t i = 0; i < k; ++i) labels.insert(distances[i].second);
    return labels;
}

}  // namespace

int main() {
    try {
        // Independently enumerable metric-boundary check.
        CountingL2Space low_level(3);
        const float left[3] = {1.0f, -2.0f, 4.0f};
        const float right[3] = {-1.0f, 1.0f, 0.0f};
        auto function = low_level.get_dist_func();
        const float distance = function(left, right, low_level.get_dist_func_param());
        require(std::fabs(distance - 29.0f) < 1e-6f, "delegated squared L2 is wrong");
        require(low_level.completed() == 1, "one completed delegate must count once");
        function(left, left, low_level.get_dist_func_param());
        require(low_level.completed() == 2, "completed calls must be additive");
        low_level.reset();
        require(low_level.completed() == 0, "reset must set the exact counter to zero");

        const std::vector<std::vector<float>> points = {
            {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {0, 3},
            {1, 3}, {2, 3}, {3, 3}, {4, 3}, {8, 8}, {-4, -4}};
        hnswlib::L2Space build_space(2);
        hnswlib::HierarchicalNSW<float> built(&build_space, points.size(), 4, 40, 2024);
        for (std::size_t i = 0; i < points.size(); ++i) {
            built.addPoint(points[i].data(), i);
        }
        const std::string path = "/tmp/coursepaper-fast-hnsw-counter-test-" +
                                 std::to_string(static_cast<long long>(getpid())) + ".hnsw";
        built.saveIndex(path);

        CountingL2Space search_space(2);
        hnswlib::HierarchicalNSW<float> loaded(&search_space, path);
        const float query_a[2] = {2.1f, 0.2f};
        const float query_b[2] = {0.1f, 2.8f};
        const std::size_t k = std::min<std::size_t>(10, points.size());

        loaded.setEf(k);
        search_space.reset();
        loaded.searchKnn(query_a, k);
        const std::uint64_t count_low_ef = search_space.completed();
        require(count_low_ef > 0, "low-ef search must execute metric evaluations");

        loaded.setEf(points.size());
        search_space.reset();
        const auto result_a = loaded.searchKnn(query_a, k);
        const std::uint64_t count_a = search_space.completed();
        require(count_a >= count_low_ef, "higher ef unexpectedly reduced the exact count");
        require(labels_of(result_a) == brute_labels(points, query_a, k),
                "tiny Recall@10 search A labels differ from brute force");

        search_space.reset();
        const auto result_b = loaded.searchKnn(query_b, k);
        const std::uint64_t count_b = search_space.completed();
        require(count_b > 0, "search B must execute metric evaluations");
        require(labels_of(result_b) == brute_labels(points, query_b, k),
                "tiny Recall@10 search B labels differ from brute force");

        search_space.reset();
        loaded.searchKnn(query_a, k);
        loaded.searchKnn(query_b, k);
        require(search_space.completed() == count_a + count_b,
                "two-query total must equal independently reset per-query totals");

        // Stored-vector inspection must not call the metric wrapper.
        const std::uint64_t before_inspection = search_space.completed();
        const std::vector<float> stored = loaded.getDataByLabel<float>(7);
        require(stored == points[7], "stored vector/label inspection is wrong");
        require(search_space.completed() == before_inspection,
                "stored-vector inspection leaked into search metric counter");

        std::remove(path.c_str());
        std::cout << "fast_hnsw_query_counter_test: PASS"
                  << " k=" << k << " low_ef_count=" << count_low_ef
                  << " count_a=" << count_a << " count_b=" << count_b << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fast_hnsw_query_counter_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
