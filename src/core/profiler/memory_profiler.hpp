#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace uta {
namespace profiler {

struct MemoryStats {
    // Global memory statistics
    size_t global_memory_loads;
    size_t global_memory_stores;
    size_t global_memory_transactions;
    float global_memory_efficiency;
    
    // Shared memory statistics
    size_t shared_memory_loads;
    size_t shared_memory_stores;
    size_t bank_conflicts;
    float shared_memory_efficiency;
    
    // Cache statistics
    float l1_cache_hit_rate;
    float l2_cache_hit_rate;
    size_t cache_line_utilization;
    
    // Memory transfer statistics
    size_t host_to_device_transfers;
    size_t device_to_host_transfers;
    double transfer_bandwidth_gbps;
    
    // Memory allocation statistics
    size_t peak_memory_usage;
    size_t current_memory_usage;
    size_t memory_allocations;
    size_t memory_frees;
};

class MemoryProfiler {
public:
    static MemoryProfiler& getInstance();

    // Start/stop profiling session
    void startProfiling();
    void stopProfiling();
    
    // Memory access tracking
    void trackMemoryAccess(void* ptr, size_t size, bool is_read);
    void trackMemoryAllocation(void* ptr, size_t size);
    void trackMemoryFree(void* ptr);
    
    // Memory transfer tracking
    void trackHostToDevice(size_t size);
    void trackDeviceToHost(size_t size);
    
    // Analysis functions
    MemoryStats getMemoryStats();
    std::vector<std::string> getOptimizationSuggestions();
    
    // Memory pattern analysis
    struct AccessPattern {
        bool is_sequential;
        bool is_strided;
        size_t stride_size;
        float coalescence_rate;
    };
    
    AccessPattern analyzeAccessPattern(const std::vector<void*>& accesses);
    
    // Memory leak detection
    struct LeakReport {
        void* address;
        size_t size;
        std::string allocation_stack;
        double time_since_allocation;
    };
    
    std::vector<LeakReport> detectLeaks();

private:
    MemoryProfiler() = default;
    
    // Internal tracking structures
    struct MemoryBlock {
        void* address;
        size_t size;
        double allocation_time;
        bool is_freed;
    };
    
    std::vector<MemoryBlock> memory_blocks_;
    MemoryStats current_stats_;
    
    // Internal helper functions
    void updateCacheStatistics(void* ptr, size_t size);
    void analyzeMemoryEfficiency();
    float calculateCoalescenceRate(const std::vector<void*>& accesses);
};

} // namespace profiler
} // namespace uta
