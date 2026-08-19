#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "vendor/hnswlib/hnswlib.h"

// A stock-hnswlib SpaceInterface wrapper used only for single-threaded query
// evaluation.  The counter is incremented after (and therefore iff) the
// delegated stock squared-L2 function completes.
class CountingL2Space final : public hnswlib::SpaceInterface<float> {
 public:
    explicit CountingL2Space(std::size_t dim) : inner_(dim) {
        params_.dim = dim;  // Must remain first: hnswlib getDataByLabel reads it.
        params_.delegate = inner_.get_dist_func();
        params_.delegate_param = inner_.get_dist_func_param();
        params_.completed = 0;
    }

    std::size_t get_data_size() override {
        return inner_.get_data_size();
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return &CountingL2Space::counted_distance;
    }

    void *get_dist_func_param() override {
        return &params_;
    }

    void reset() noexcept {
        params_.completed = 0;
    }

    std::uint64_t completed() const noexcept {
        return params_.completed;
    }

 private:
    struct Params {
        // Keep dimension first for compatibility with hnswlib v0.8.0's
        // getDataByLabel implementation, which treats dist_func_param_ as a
        // pointer to size_t.
        std::size_t dim;
        hnswlib::DISTFUNC<float> delegate;
        void *delegate_param;
        std::uint64_t completed;
    };

    static float counted_distance(const void *left, const void *right, const void *opaque) {
        Params *params = const_cast<Params *>(static_cast<const Params *>(opaque));
        const float value = params->delegate(left, right, params->delegate_param);
        ++params->completed;  // Deliberately after the completed delegate call.
        return value;
    }

    hnswlib::L2Space inner_;
    Params params_{};
};
