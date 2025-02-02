#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace uta {
namespace profiler {

struct PTXMetrics {
    size_t instruction_count;
    size_t memory_loads;
    size_t memory_stores;
    size_t shared_memory_ops;
    size_t arithmetic_ops;
    size_t control_flow_ops;
    double theoretical_occupancy;
    size_t register_usage;
    size_t shared_memory_usage;
};

struct KernelMetrics {
    std::string kernel_name;
    double execution_time_ms;
    size_t grid_size[3];
    size_t block_size[3];
    size_t dynamic_shared_memory;
    PTXMetrics ptx_metrics;
};

class PTXProfiler {
public:
    static PTXProfiler& getInstance();

    // Analyze PTX code
    PTXMetrics analyzePTX(const std::string& ptx_code);

    // Profile kernel execution
    KernelMetrics profileKernel(const std::string& kernel_name,
                               void* kernel_func,
                               const void** arguments,
                               size_t num_arguments,
                               const dim3& grid_dim,
                               const dim3& block_dim,
                               size_t shared_memory = 0);

    // Get optimization suggestions
    std::vector<std::string> getOptimizationSuggestions(const KernelMetrics& metrics);

    // Memory access pattern analysis
    struct MemoryAccessPattern {
        bool is_coalesced;
        bool has_bank_conflicts;
        float l1_hit_rate;
        float l2_hit_rate;
    };
    
    MemoryAccessPattern analyzeMemoryPattern(const std::string& ptx_code);

private:
    PTXProfiler() = default;

    // Internal analysis functions
    size_t countInstructions(const std::string& ptx_code);
    size_t analyzeRegisterUsage(const std::string& ptx_code);
    size_t analyzeSharedMemoryUsage(const std::string& ptx_code);
    double calculateTheoreticalOccupancy(const PTXMetrics& metrics);
    
    // Cache for analysis results
    std::unordered_map<std::string, PTXMetrics> metrics_cache_;
};

} // namespace profiler
} // namespace uta
