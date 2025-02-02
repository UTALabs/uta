#pragma once

#include <vector>
#include <string>
#include <memory>
#include "../profiler/memory_profiler.hpp"

namespace uta {
namespace autotuner {

struct MemoryAccessPattern {
    enum class Type {
        SEQUENTIAL,
        STRIDED,
        RANDOM,
        COALESCED,
        BROADCAST
    };
    
    Type pattern_type;
    size_t stride_size;
    float coalescence_rate;
    bool has_bank_conflicts;
};

struct MemoryConfig {
    size_t cache_line_size;
    size_t warp_size;
    size_t shared_memory_banks;
    size_t l1_cache_size;
    size_t l2_cache_size;
};

class MemoryOptimizer {
public:
    static MemoryOptimizer& getInstance();

    // Memory access pattern optimization
    struct AccessOptimization {
        std::string ptx_code;
        std::vector<std::string> optimization_steps;
        float estimated_improvement;
    };
    
    AccessOptimization optimizeAccessPattern(
        const std::string& original_ptx,
        const MemoryAccessPattern& current_pattern
    );
    
    // Shared memory optimization
    struct SharedMemoryConfig {
        size_t block_size;
        size_t elements_per_thread;
        size_t padding_size;
        bool use_double_buffering;
    };
    
    SharedMemoryConfig optimizeSharedMemory(
        size_t data_size,
        const MemoryAccessPattern& access_pattern
    );
    
    // Cache optimization
    struct CacheStrategy {
        bool prefer_l1_cache;
        bool use_texture_cache;
        size_t prefetch_distance;
        std::vector<std::string> cache_hints;
    };
    
    CacheStrategy optimizeCacheUsage(
        const MemoryAccessPattern& access_pattern,
        const MemoryConfig& memory_config
    );

private:
    MemoryOptimizer() = default;
    
    // Internal optimization algorithms
    std::string generateCoalescedAccess(
        const std::string& ptx_code,
        const MemoryAccessPattern& pattern
    );
    
    std::string optimizeBankConflicts(
        const std::string& ptx_code,
        size_t shared_memory_banks
    );
    
    std::string insertPrefetchInstructions(
        const std::string& ptx_code,
        const CacheStrategy& strategy
    );
    
    // Helper functions
    bool analyzeMemoryDivergence(const std::string& ptx_code);
    float estimateMemoryThroughput(const MemoryAccessPattern& pattern);
    std::vector<std::string> generateMemoryHints(const CacheStrategy& strategy);
};

} // namespace autotuner
} // namespace uta
