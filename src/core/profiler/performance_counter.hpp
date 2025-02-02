#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>

namespace uta {
namespace profiler {

// Counter type
enum class CounterType {
    CYCLES,             // CPU cycle
    INSTRUCTIONS,       // instruction count
    CACHE_MISSES,      // cache miss
    BRANCH_MISSES,     // branch prediction miss
    MEMORY_LOADS,      // memory load
    MEMORY_STORES,     // memory store
    FLOPS,             // floating point operation
    CUSTOM             // custom counter
};

// Counter configuration
struct CounterConfig {
    CounterType type;
    std::string name;
    bool enabled;
    uint64_t sampling_rate;
    uint64_t threshold;
};

// Counter event
struct CounterEvent {
    CounterType type;
    std::string name;
    uint64_t value;
    std::chrono::steady_clock::time_point timestamp;
    int device_id;
    std::string context;
};

// Performance counter
class PerformanceCounter {
public:
    static PerformanceCounter& getInstance();

    // Initialization
    void initialize(const std::vector<CounterConfig>& configs);

    // Counter operations
    void start(const std::string& name);
    void stop(const std::string& name);
    void increment(const std::string& name, uint64_t value = 1);
    void reset(const std::string& name);
    uint64_t getValue(const std::string& name) const;

    // Sampling control
    void enableSampling(const std::string& name, uint64_t rate);
    void disableSampling(const std::string& name);

    // Event handling
    void addEventHandler(std::function<void(const CounterEvent&)> handler);
    void removeEventHandler(size_t handler_id);

    // Statistics
    struct CounterStats {
        uint64_t total;
        uint64_t min;
        uint64_t max;
        double average;
        double standard_deviation;
        std::vector<CounterEvent> history;
    };

    CounterStats getStats(const std::string& name) const;
    std::vector<CounterStats> getAllStats() const;

private:
    PerformanceCounter() = default;

    // internal state
    struct CounterState {
        std::atomic<uint64_t> value;
        uint64_t start_value;
        std::chrono::steady_clock::time_point start_time;
        bool running;
        CounterConfig config;
        std::vector<CounterEvent> events;
    };

    std::unordered_map<std::string, CounterState> counters_;
    std::vector<std::function<void(const CounterEvent&)>> event_handlers_;
    mutable std::mutex mutex_;

    // Sampling control
    bool shouldSample(const std::string& name);
    void recordEvent(const std::string& name, uint64_t value);
};

// Hardware counter access
class HardwareCounterAccess {
public:
    static HardwareCounterAccess& getInstance();

    // Initialization
    bool initialize();

    // Counter operations
    uint64_t readCounter(CounterType type);
    void resetCounter(CounterType type);
    bool isCounterSupported(CounterType type) const;

    // Hardware information
    struct HardwareInfo {
        std::vector<CounterType> supported_counters;
        size_t max_counters;
        std::string cpu_name;
        std::string architecture;
    };

    HardwareInfo getHardwareInfo() const;

private:
    HardwareCounterAccess() = default;

    // Internal functions
    void setupCounters();
    void cleanupCounters();
    uint64_t readMSR(uint32_t msr) const;
    void writeMSR(uint32_t msr, uint64_t value);

    // Hardware state
    HardwareInfo hw_info_;
    bool initialized_{false};
};

// CUDA performance counter
class CUDAPerformanceCounter {
public:
    static CUDAPerformanceCounter& getInstance();

    // Initialization
    void initialize(int device_id);

    // CUDA specific counter
    enum class CUDACounterType {
        SM_OCCUPANCY,           // SM occupancy
        MEMORY_THROUGHPUT,      // memory throughput
        CACHE_HIT_RATE,        // cache hit rate
        WARP_EXECUTION_EFFICIENCY, // Warp execution efficiency
        INSTRUCTION_THROUGHPUT,    // instruction throughput
        MEMORY_BANDWIDTH          // memory bandwidth
    };

    // Counter operations
    void startCounter(CUDACounterType type);
    void stopCounter(CUDACounterType type);
    double getValue(CUDACounterType type) const;

    // Performance metrics
    struct CUDAPerformanceMetrics {
        double sm_efficiency;
        double memory_efficiency;
        double instruction_efficiency;
        double bandwidth_utilization;
    };

    CUDAPerformanceMetrics getMetrics() const;

private:
    CUDAPerformanceCounter() = default;

    // internal state
    int device_id_;
    std::unordered_map<CUDACounterType, uint64_t> counter_values_;
    bool initialized_{false};
};

} // namespace profiler
} // namespace uta
