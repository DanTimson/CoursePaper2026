#include "extension.h"
#include "test_config.h"
#include "test_readfile.h"
#include "baseline.h"
#include "baseline2.h"
#include "memory-optimize.h"
// #include "memory-optimize-limit2G.h"
#include <cstdint> // For int64_t
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream> // For stringstream
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
using namespace std;

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
            if (id % 200000 == 0) {
                std::cout << "Processed " << id << " items." << std::endl;
            }
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                        if (id % 200000 == 0) {
                            std::cout << "Processed " << id << " items." << std::endl;
                        }
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}
float global_counter = 0.0f;
struct mergedIndex {
    size_t count;
    hnswlib::HierarchicalNSW<float> *index;
};
bool compareByCountDesc(const mergedIndex &a, const mergedIndex &b) {
    return a.count >= b.count;
}
bool compareByCountAsc(const mergedIndex &a, const mergedIndex &b) {
    return a.count <= b.count;
}

void workload(const std::string &config_path) {
    Config cfg = loadConfig(config_path);
    size_t dim = cfg.dim;
    size_t max_elements = cfg.max_elements;
    size_t nb = cfg.nb;
    int M = cfg.M;
    int ef_construction = cfg.ef_construction;
    int iterations = cfg.iterations;
    int kk = cfg.kk;
    int k = cfg.k;
    int nq = cfg.nq;
    int lrange = cfg.lrange;
    int rrange = cfg.rrange;
    int thread = cfg.thread;
    int lambda = cfg.lambda;
    size_t jump_ef = (size_t)cfg.jump_ef;
    size_t local_ef = (size_t)cfg.local_ef;
    size_t next_step_k = (size_t)cfg.next_step_k;
    size_t next_step_ef = (size_t)cfg.next_step_ef;
    size_t search_M = (size_t)cfg.search_M;
    size_t search_ef = (size_t)cfg.search_ef;
    int merge_ef_construction = cfg.merge_ef_construction;
    float alpha = cfg.alpha;
    bool save_index = cfg.save_index;

    printf("Configuration:\n");
    printf("  Workload Type: %s\n", workloadTypeToString(cfg.workload_type).c_str());
    printf("  Merge Method: %s\n", mergeMethodToString(cfg.merge_method).c_str());
    printf("  Dimension: %ld\n", dim);
    printf("  Max Elements: %ld\n", max_elements);
    printf("  Base Vectors: %ld\n", nb);
    printf("  M: %d\n", M);
    printf("  ef_construction: %d\n", ef_construction);
    printf("  k: %d\n", k);
    printf("  Query Vectors: %d\n", nq);
    printf("  Iterations: %d\n", iterations);
    printf("  Rerun: %s\n", cfg.rerun ? "true" : "false");
    printf("  Thread Count: %d\n", thread);
    printf("  lambda: %d\n", lambda);
    printf("  Alpha: %.2f\n", alpha);
    printf("  Save Index: %s\n", save_index ? "true" : "false");

    enum WorkloadType workload_type = cfg.workload_type;
    enum MergeMethod merge_method = cfg.merge_method;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = nullptr;
    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = nullptr;
    hnswlib::HierarchicalNSW<float> *alg_hnsw2 = nullptr;

    float *xb = new float[dim * nb];
    size_t dd2 = dim;
    size_t nt2 = nb;

    float *xq = new float[dim * nq];
    size_t dd = dim;
    size_t nt = nq;
    xq = read_vectors(const_cast<char *>(cfg.query_filepath.c_str()), nq, &dd, &nt);
    printf("loaded query vectors: %zu vectors of dimension %zu\n", nt, dd);

    size_t k_file = kk;
    size_t nqq = nq;
    int *gt_int = ivecs_read(const_cast<char *>(cfg.groundtruth_filepath.c_str()), &k_file, &nqq);
    int *gt = new int[nq * k];
    if (k_file > k) {
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                gt[i * k + j] = gt_int[i * kk + j];
            }
        }
    } else {
        gt = gt_int;
    }
    printf("loaded ground-truth for %d queries, top %d\n", nq, k);

    printf("Merge method: %s\n", mergeMethodToString(merge_method).c_str());
    if (merge_method == REBUILD) {
        if (cfg.rerun) {
            xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
            printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                delete alg_hnsw0;
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                double t0 = elapsed();
                ParallelFor(lrange, rrange, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string index_path = cfg.index_path[0];
                    alg_hnsw0->saveIndex(index_path);
                    printf("Saved index to: %s\n", index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
        } else {
            std::string index_path = cfg.index_path[0];
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path, &space);
            printf("Loaded merged index from: %s\n", index_path.c_str());
        }
    } else if (merge_method == INSERT) {
        if (cfg.rerun) {
            xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
            printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                std::string index_path = (cfg.index_path)[0];
                printf("Loading index from: %s\n", index_path.c_str());
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                alg_hnsw0->loadIndex(index_path, &space, max_elements);
                std::cout << "Loaded index from: " << index_path << std::endl;
                if (merge_ef_construction > 0) {   // SIGM effort knob
                    alg_hnsw0->ef_construction_ = (size_t)merge_ef_construction;
                    printf("SIGM ef_construction = %d\n", merge_ef_construction);
                }

                double t0 = elapsed();
                ParallelFor(lrange, rrange, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw0->get_dist_call_counter());
            }
        }
        printf("Insert task do not re-test the performance.\n");
        return;
    } else if (merge_method == TWO_MERGE) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw0, alg_hnsw1, &space, lambda);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw2->get_dist_call_counter());   // HNSWMerger patch: instrument the SIGMOD merge (was time-only)
                if (save_index) {
                    std::string merged_index_path = save_base + "/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == MEMORY_EFFICIENCY) {
        // throw std::runtime_error("Memory efficiency mode is deprecated, please use MEMORY_EFFICIENCY_LIMIT instead.");
        if (save_index == false) {
            std::runtime_error("Memory efficiency mode test always need to write index to disk");
        }
        std::string save_base = cfg.save_path;
        std::string merged_index_path = save_base + "/lessmem-index_" + workloadTypeToString(workload_type) + ".hnsw";
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                hnswlib::HNSWMerger_ME<float>(index_path[0], index_path[1], merged_index_path, &space, max_elements, M, ef_construction);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("Saved merged index to: %s\n", merged_index_path.c_str());
                // exit(0);
            }
        }
        alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        alg_hnsw0->loadIndex(merged_index_path, &space);
        printf("Loaded merged index from: %s\n", merged_index_path.c_str());
    } else if (merge_method == MEMORY_EFFICIENCY_LIMIT) {
        throw std::runtime_error("Memory efficiency limit mode is deprecated, please use MEMORY_EFFICIENCY_LIMIT2G instead.");
        // // NOTE: The deleting mark design in this function has bugs. This will not affect the performance of the merging under limit, but only need to ignore the delete detection when loading merged index.
        // if (save_index == false) {
        //     std::runtime_error("Memory efficiency mode test always need to write index to disk");
        // }
        // std::string save_base = cfg.save_path;
        // std::string merged_index_path = save_base + "/lessmem-index_" + workloadTypeToString(workload_type) + ".hnsw";
        // if (cfg.rerun == true) {
        //     std::vector<std::string> index_path = cfg.index_path;
        //     for (int i = 0; i < iterations; i++) {
        //         printf("Iteration %d/%d\n", i + 1, iterations);

        //         double t0 = elapsed();
        //         hnswlib::HNSWMerger_ME<float>(index_path[0], index_path[1], merged_index_path, &space, max_elements, M, ef_construction);
        //         printf("Total time for insertion: %.3f s\n", elapsed() - t0);
        //         printf("Saved merged index to: %s\n", merged_index_path.c_str());
        //         exit(0);
        //     }
        // }
        // alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        // alg_hnsw0->loadIndex(merged_index_path, &space);
        // printf("Loaded merged index from: %s\n", merged_index_path.c_str());
    } else if (merge_method == BACKWARD_SEARCH) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_BS<float>(alg_hnsw0, alg_hnsw1, &space, lambda);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = save_base + "/backward-search_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/backward-search_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == MULTI_TWO_MERGE) {
        printf("multi_merge_method: %s\n", multiTestMethodToString(cfg.multi_test_method).c_str());
        std::string save_base = cfg.save_path;
        enum MultiTestMethod multi_test_method = cfg.multi_test_method;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            std::vector<mergedIndex> indexVec;
            for (int i = index_path.size() - 1; i >= 0; i--) {
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                alg_hnsw0->loadIndex(index_path[i], &space);
                printf("Loaded index from: %s\n", index_path[i].c_str());
                indexVec.push_back({alg_hnsw0->getCurrentElementCount(), alg_hnsw0});
            }
            if (multi_test_method == LARGE_FIRST) {
                std::sort(indexVec.begin(), indexVec.end(), compareByCountDesc);
            } else if (multi_test_method == SMALL_FIRST) {
                std::sort(indexVec.begin(), indexVec.end(), compareByCountAsc);
            } else if (multi_test_method == RANDOM) {
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(indexVec.begin(), indexVec.end(), g);
            } else {
                printf("Unknown multi test method, defaulting to LARGE_FIRST.\n");
                exit(1);
            }
            std::vector<mergedIndex> indexVec2(indexVec);

            for (int i = 0; i < iterations; i++) {
                printf("Sorted indexes by count:\n");
                indexVec.clear();
                indexVec.insert(indexVec.begin(), indexVec2.begin(), indexVec2.end());
                double t0 = elapsed();
                printf("Iteration %d/%d\n", i + 1, iterations);
                // int delta = 4, lambda = 8;
                size_t baseline = indexVec[0].count;
                for (int it = 1; it < indexVec.size(); it++) {
                    baseline = std::min(baseline, indexVec[it].count);
                }
                // int upper_bound = 16;
                while (indexVec.size() > 1) {
                    mergedIndex first = indexVec[0];
                    mergedIndex second = indexVec[1];
                    indexVec.erase(indexVec.begin(), indexVec.begin() + 2);
                    printf("Merging indexes with counts: %zu and %zu\n", first.count, second.count);
                    // lambda = static_cast<int>(4 * std::log(std::max(first.count, second.count) / 693000.0) / std::log(1000000.0 / 693000.0));
                    // lambda = 4;

                    printf("lambda: %d\n", lambda);
                    alg_hnsw2 = hnswlib::HNSWMerger<float>(first.index, second.index, &space, lambda);
                    first.index->clear();
                    second.index->clear();
                    printf("Merged index with counts: %zu, time elapsed: %f s\n", first.count + second.count, elapsed() - t0);
                    if (multi_test_method == LARGE_FIRST) {
                        indexVec.insert(indexVec.begin(), {alg_hnsw2->getCurrentElementCount(), alg_hnsw2});
                    } else if (multi_test_method == SMALL_FIRST) {
                        indexVec.push_back({alg_hnsw2->getCurrentElementCount(), alg_hnsw2});
                        std::sort(indexVec.begin(), indexVec.end(), compareByCountAsc);
                    } else if (multi_test_method == RANDOM) {
                        indexVec.push_back({alg_hnsw2->getCurrentElementCount(), alg_hnsw2});
                        std::random_device rd;
                        std::mt19937 g(rd());
                        std::shuffle(indexVec.begin(), indexVec.end(), g);
                    }
                }

                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("Total global counter: %.3f\n", global_counter);

                if (save_index) {
                    std::string merged_index_path = save_base + "/multi_" + multiTestMethodToString(multi_test_method) + "_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/multi_" + multiTestMethodToString(multi_test_method) + "_" + workloadTypeToString(workload_type) + ".hnsw";

            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == ABLATION_C) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw0, alg_hnsw1, &space, lambda, alpha);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = save_base + "/ablation_c_" + std::to_string(lambda) + "_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/ablation_c_" + std::to_string(lambda) + "_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == ES) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_ES<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw2->get_dist_call_counter());

                if (save_index) {
                    std::string merged_index_path = save_base + "/es_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/es_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == NGM) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                // size_t search_ef = 20;
                // alg_hnsw2 = hnswlib::HNSWMerger_Naive<float>(alg_hnsw0, alg_hnsw1, &space, search_ef);
                alg_hnsw2 = hnswlib::HNSWMerger_Naive<float>(alg_hnsw0, alg_hnsw1, &space, search_ef);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw2->get_dist_call_counter());
                if (save_index) {
                    std::string merged_index_path = save_base + "/ngm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/ngm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == IGTM) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                // size_t jump_ef = 20;
                // size_t local_ef = 5;
                // size_t next_step_k = 3;
                // size_t next_step_ef = 3;
                // size_t search_M = 20;
                // alg_hnsw2 = hnswlib::HNSWMerger_IGTM<float>(alg_hnsw0, alg_hnsw1, &space, jump_ef, local_ef, next_step_k, next_step_ef, search_M);
                alg_hnsw2 = hnswlib::HNSWMerger_IGTM<float>(alg_hnsw0, alg_hnsw1, &space, jump_ef, local_ef, next_step_k, next_step_ef, search_M);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw2->get_dist_call_counter());
                if (save_index) {
                    std::string merged_index_path = save_base + "/igtm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saving merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/igtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == CGTM) {
        std::string save_base = cfg.save_path;
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                // size_t jump_ef = 20;
                // size_t local_ef = 5;
                // size_t next_step_k = 3;
                // size_t search_M = 20;
                alg_hnsw2 = hnswlib::HNSWMerger_CGTM<float>(alg_hnsw0, alg_hnsw1, &space, jump_ef, local_ef, next_step_k, search_M);
                // alg_hnsw2 = hnswlib::HNSWMerger_CGTM<float>(alg_hnsw0, alg_hnsw1, &space, jump_ef, local_ef, next_step_k, search_M);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                printf("distance calls = %llu\n", (unsigned long long)alg_hnsw2->get_dist_call_counter());
                if (save_index) {
                    std::string merged_index_path = save_base + "/cgtm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saving merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = save_base + "/cgtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else {
        std::cerr << "Unknown merge method!" << std::endl;
        exit(1);
    }

    printf("Start searching\n");
    for (int ef_val : cfg.efs_array) {
        alg_hnsw0->setEf(ef_val) ;
        printf("set ef = %d\n", ef_val);
        for (int iter = 0; iter < 3; iter++) {
            double t_search = 0.0;
            double t0 = elapsed();
            unsigned long long dc0 = (unsigned long long)alg_hnsw0->get_dist_call_counter();   // d_s patch
            int *I = new int[nq * k];
            for (int i = 0; i < nq; i++) {
                double t1 = elapsed();
                auto result = alg_hnsw0->searchKnn(xq + i * dim, k);
                t_search += elapsed() - t1;
                for (int j = k - 1; j >= 0; j--) {
                    I[i * k + j] = static_cast<int>(result.top().second);
                    result.pop();
                }
            }
            double t_all = elapsed() - t0;
            double d_s = (double)((unsigned long long)alg_hnsw0->get_dist_call_counter() - dc0) / (double)nq;   // d_s patch
            printf("[search time: %.3f s, pure query time: %.3f s] ef=%d\n", t_all, t_search, ef_val);
            printf("search distance calls per query = %.4f\n", d_s);   // d_s patch: iso-quality metric

            int correct = 0;
            for (int i = 0; i < nq; i++) {
                std::map<int, int> umap;
                for (int j = 0; j < k; j++) {
                    umap[gt[i * k + j]] = 1;
                }
                for (int j = 0; j < k; j++) {
                    if (umap.find(I[i * k + j]) != umap.end()) {
                        correct++;
                    }
                }
            }
            float recall = correct / static_cast<float>(nq * k);
            printf("Intersection-merged index R@100 = %.4f\n\n", recall);

            delete[] I;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    workload(config_path);
    return 0;
}