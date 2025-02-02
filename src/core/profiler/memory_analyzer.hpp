#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace uta {
namespace profiler {

// memory access mode
enum class MemoryAccessPattern {
    SEQUENTIAL,          // sequential access
    STRIDED,            // strided access
    RANDOM,             // random access
    COALESCED,          // coalesced access
    CONFLICTING         // conflicting access
};

// memory event type
enum class MemoryEventType {
    ALLOCATION,         // memory allocation
    DEALLOCATION,      // memory deallocation
    READ,              // read operation
    WRITE,             // write operation
    COPY,              // copy operation
    PAGE_FAULT         // page fault
};

// memory event
struct MemoryEvent {
    MemoryEventType type;
    void* address;
    size_t size;
    std::chrono::steady_clock::time_point timestamp;
    int device_id;
    std::string context;
    MemoryAccessPattern pattern;
};

// memory analyzer
class MemoryAnalyzer {
public:
    static MemoryAnalyzer& getInstance();

    // initialization
    void initialize();

    // event log
    void recordEvent(const MemoryEvent& event);

    // access pattern analysis
    MemoryAccessPattern analyzeAccessPattern(
        const std::vector<void*>& addresses,
        const std::vector<size_t>& sizes
    );

    // Memory Leak Detection
    struct LeakInfo {
        void* address;
        size_t size;
        std::string allocation_context;
        std::chrono::steady_clock::time_point allocation_time;
    };

    std::vector<LeakInfo> detectLeaks();

    // memory statistics
    struct MemoryStats {
        size_t total_allocated;
        size_t peak_allocated;
        size_t current_allocated;
        size_t allocation_count;
        size_t deallocation_count;
        std::unordered_map<MemoryAccessPattern, size_t> access_patterns;
    };

    MemoryStats getStats() const;

private:
    MemoryAnalyzer() = default;

    // internal state
    struct AllocationInfo {
        size_t size;
        std::string context;
        std::chrono::steady_clock::time_point timestamp;
        bool freed;
    };

    std::unordered_map<void*, AllocationInfo> allocations_;
    MemoryStats stats_;
    
    // analytical function
    void updateStats(const MemoryEvent& event);
    void checkForLeaks();
    void analyzeFragmentation();
};

// cache analyzer
class CacheAnalyzer {
public:
    static CacheAnalyzer& getInstance();

    // cache configuration
    struct CacheConfig {
        size_t line_size;
        size_t cache_size;
        size_t associativity;
        size_t num_sets;
    };

    void initialize(const CacheConfig& config);

    // cache emulation
    void simulateAccess(void* address, size_t size, bool is_write);

    // cache statistics
    struct CacheStats {
        size_t hits;
        size_t misses;
        double hit_rate;
        size_t evictions;
        size_t writebacks;
    };

    CacheStats getStats() const;

private:
    CacheAnalyzer() = default;

    // cache line
    struct CacheLine {
        void* tag;
        bool valid;
        bool dirty;
        size_t last_access;
    };

    // cache set organization
    std::vector<std::vector<CacheLine>> cache_sets_;
    CacheConfig config_;
    CacheStats stats_;

    // cache operations
    size_t getCacheSet(void* address);
    void* getTag(void* address);
    void updateCache(size_t set_index, void* tag, bool is_write);
};

// bandwidth analyzer
class BandwidthAnalyzer {
public:
    static BandwidthAnalyzer& getInstance();

    // initialization
    void initialize();

    // bandwidth measurement
    void recordTransfer(
        size_t size,
        std::chrono::duration<double> duration,
        bool is_read
    );

    // bandwidth statistics
    struct BandwidthStats {
        double peak_bandwidth;
        double average_bandwidth;
        double current_bandwidth;
        size_t total_bytes_transferred;
        std::chrono::duration<double> total_transfer_time;
    };

    BandwidthStats getStats() const;

    // Bottleneck analysis
    struct BottleneckInfo {
        bool is_bandwidth_bound;
        bool is_latency_bound;
        double bandwidth_utilization;
        double latency_impact;
    };

    BottleneckInfo analyzeBottlenecks();

private:
    BandwidthAnalyzer() = default;

    // internal state
    struct TransferInfo {
        size_t size;
        std::chrono::duration<double> duration;
        bool is_read;
        double bandwidth;
    };

    std::vector<TransferInfo> transfers_;
    BandwidthStats stats_;

    // analytical function
    void updateStats(const TransferInfo& transfer);
    double calculateBandwidth(size_t size, std::chrono::duration<double> duration);
};

} // namespace profiler
} // namespace uta
