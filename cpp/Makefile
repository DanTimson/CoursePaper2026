CXX ?= g++
CPPFLAGS ?= -I.
CXXFLAGS ?= -O3 -DNDEBUG -std=c++11 -Wall -Wextra -Wpedantic
LDLIBS ?= -pthread
FASTKCNA_ROOT ?= ../FastKCNA
FASTKCNA_INCLUDE ?= $(FASTKCNA_ROOT)/code/include
FASTKCNA_LIB ?= $(FASTKCNA_ROOT)/code/libkgraph.a

.PHONY: all layerwise test clean

all: fast_hnsw_quality_eval fast_hnsw_query_counter_test

fast_hnsw_quality_eval: fast_hnsw_quality_eval.cpp fast_hnsw_counting_space.h $(wildcard vendor/hnswlib/*.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

fast_hnsw_query_counter_test: fast_hnsw_query_counter_test.cpp fast_hnsw_counting_space.h $(wildcard vendor/hnswlib/*.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)


layerwise: layerwise_nnd_hnsw_builder layerwise_nnd_hnsw_validate

layerwise_nnd_hnsw_builder: layerwise_nnd_hnsw_builder.cpp $(FASTKCNA_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp -I$(FASTKCNA_INCLUDE) $< $(FASTKCNA_LIB) -o $@ -lrt

layerwise_nnd_hnsw_validate: layerwise_nnd_hnsw_validate.cpp $(wildcard vendor/hnswlib/*.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

test: fast_hnsw_query_counter_test
	./fast_hnsw_query_counter_test

clean:
	rm -f fast_hnsw_quality_eval fast_hnsw_query_counter_test layerwise_nnd_hnsw_builder layerwise_nnd_hnsw_validate
