#pragma once

#include "uta.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <functional>

namespace uta {
namespace profiler {

// profiling configuration
struct ProfilerConfig {
    bool enabled;
    bool record_shapes;
    bool record_memory;
    bool record_bandwidth;
    bool record_flops;
    std::string output_dir;
};

// performance metrics
struct Metrics {
    double execution_time;      // Execution time (ms)
    double memory_used;         // Memory used (bytes)
    double bandwidth;           // Bandwidth utilization (GB/s)
    double flops;              // Floating point operations
    double flops_per_second;   // FLOPS
    double occupancy;          // SM occupancy
};

// operational statistics
struct OperationStats {
    std::string name;
    std::string type;
    Metrics metrics;
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<std::vector<size_t>> output_shapes;
};

// memory event
struct MemoryEvent {
    enum class Type {
        ALLOC,
        FREE,
        H2D,
        D2H,
        D2D
    };
    
    Type type;
    size_t size;
    void* ptr;
    std::chrono::steady_clock::time_point timestamp;
};

// performance analyzer
class Profiler {
public:
    static Profiler& getInstance();
    
    // configuration
    void configure(const ProfilerConfig& config);
    
    // control
    void start();
    void stop();
    void reset();
    
    // record operation     
    void recordOperation(
        const std::string& name,
        const std::function<void()>& op,
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs
    );
    
    // record event
    void recordMemoryEvent(const MemoryEvent& event);
    
    // get statistics
    std::vector<OperationStats> getStats();
    
    // generate report
    
    // visualize
    void visualizeTimeline(const std::string& filename);
    void visualizeMemoryUsage(const std::string& filename);
    void visualizeOperationGraph(const std::string& filename);
};

// performance scope
class ScopedProfile {
public:
    ScopedProfile(const std::string& name);
    ~ScopedProfile();
    
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_time_;
};

// macro definition
#define UTA_PROFILE_SCOPE(name) \
    uta::profiler::ScopedProfile __profile_##__LINE__(name)

#define UTA_PROFILE_FUNCTION() \
    UTA_PROFILE_SCOPE(__FUNCTION__)

} // namespace profiler
} // namespace uta
