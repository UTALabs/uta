#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include "ptx_profiler.hpp"
#include "memory_profiler.hpp"

namespace uta {
namespace profiler {

struct KernelProfile {
    // Basic information
    std::string kernel_name;
    std::string ptx_version;
    
    // Execution statistics
    double total_time_ms;
    double average_time_ms;
    size_t num_calls;
    
    // Resource utilization
    struct ResourceUsage {
        size_t registers_per_thread;
        size_t shared_memory_per_block;
        size_t constant_memory_size;
        double sm_occupancy;
    } resource_usage;
    
    // Performance metrics
    struct Performance {
        double flops;
        double bandwidth_gbps;
        double instruction_throughput;
        double warp_execution_efficiency;
    } performance;
    
    // Thread configuration
    struct ThreadConfig {
        dim3 grid_dim;
        dim3 block_dim;
        size_t dynamic_shared_memory;
    } thread_config;
};

class KernelProfiler {
public:
    static KernelProfiler& getInstance();

    // Profiling control
    void startProfiling(const std::string& kernel_name);
    void stopProfiling();
    
    // Profile collection
    KernelProfile getProfile(const std::string& kernel_name);
    std::vector<KernelProfile> getAllProfiles();
    
    // Analysis and optimization
    struct OptimizationSuggestion {
        std::string description;
        std::string impact;
        std::string implementation_hint;
        float expected_improvement;
    };
    
    std::vector<OptimizationSuggestion> analyzeKernel(const std::string& kernel_name);
    
    // Thread configuration optimization
    struct OptimalConfig {
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_memory;
        float estimated_performance;
    };
    
    OptimalConfig findOptimalConfig(const std::string& kernel_name,
                                  const std::vector<dim3>& candidate_grids,
                                  const std::vector<dim3>& candidate_blocks);

private:
    KernelProfiler() = default;
    
    // Internal profiling state
    struct ProfilingState {
        std::chrono::high_resolution_clock::time_point start_time;
        std::string current_kernel;
        bool is_profiling;
    };
    
    ProfilingState state_;
    std::unordered_map<std::string, KernelProfile> profiles_;
    
    // Analysis helpers
    double calculateFLOPs(const std::string& ptx_code);
    double calculateMemoryBandwidth(const MemoryStats& mem_stats);
    double calculateOccupancy(const ResourceUsage& resources);
    
    // Optimization helpers
    std::vector<OptimizationSuggestion> analyzePTXOptimizations(const std::string& ptx_code);
    std::vector<OptimizationSuggestion> analyzeMemoryOptimizations(const MemoryStats& mem_stats);
    std::vector<OptimizationSuggestion> analyzeResourceUtilization(const ResourceUsage& resources);
};

} // namespace profiler
} // namespace uta
