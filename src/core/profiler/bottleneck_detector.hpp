#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "performance_counter.hpp"
#include "memory_analyzer.hpp"

namespace uta {
namespace profiler {

// Bottleneck type
enum class BottleneckType {
    COMPUTE_BOUND,      // compute bound
    MEMORY_BOUND,       // memory bound
    BANDWIDTH_BOUND,    // bandwidth bound
    LATENCY_BOUND,      // latency bound
    SYNCHRONIZATION,    // synchronization bound
    LOAD_IMBALANCE     // load imbalance
};

// performance metrics
struct PerformanceMetrics {
    double flop_rate;           // FLOP/s
    double memory_bandwidth;    // GB/s
    double cache_hit_rate;      // cache hit rate
    double sm_occupancy;        // SM occupancy
    double load_balance;        // load balance
    double sync_overhead;       // sync overhead
};

// Bottleneck detector
class BottleneckDetector {
public:
    static BottleneckDetector& getInstance();

    // initialization
    void initialize();

    // performance analysis
    struct PerformanceAnalysis {
        std::vector<BottleneckType> detected_bottlenecks;
        std::unordered_map<BottleneckType, double> bottleneck_severity;
        std::vector<std::string> optimization_suggestions;
        PerformanceMetrics metrics;
    };

    PerformanceAnalysis analyzePerformance();

    // Bottleneck detection configuration
    struct DetectionConfig {
        double compute_threshold;    // compute density threshold
        double memory_threshold;     // memory access threshold
        double bandwidth_threshold;  // bandwidth utilization threshold
        double latency_threshold;    // latency sensitivity threshold
        double sync_threshold;       // synchronization overhead threshold
        double imbalance_threshold;  // load imbalance threshold
    };

    void setConfig(const DetectionConfig& config);

private:
    BottleneckDetector() = default;

    // analytical function
    void analyzeComputeEfficiency();
    void analyzeMemoryEfficiency();
    void analyzeBandwidthUtilization();
    void analyzeLatencyImpact();
    void analyzeSynchronization();
    void analyzeLoadBalance();

    // optimization suggestions
    std::vector<std::string> generateOptimizationSuggestions(
        const std::vector<BottleneckType>& bottlenecks
    );

    // internal state
    DetectionConfig config_;
    PerformanceAnalysis current_analysis_;
};

// Roofline Analyzer
class RooflineAnalyzer {
public:
    static RooflineAnalyzer& getInstance();

    // initialization
    void initialize();

    // Roofline model configuration
    struct RooflineConfig {
        double peak_compute;     // Peak FLOP/s
        double peak_bandwidth;   // Peak GB/s
        size_t cache_size;      // Cache size
        double cache_bandwidth; // Cache bandwidth
    };

    void setConfig(const RooflineConfig& config);

    // performance analysis
    struct RooflinePoint {
        double arithmetic_intensity;  // FLOP/byte
        double performance;          // FLOP/s
        std::string kernel_name;
    };

    void addDataPoint(const RooflinePoint& point);
    std::vector<RooflinePoint> getDataPoints() const;

    // performance boundary analysis
    struct PerformanceBounds {
        double compute_bound;
        double memory_bound;
        double cache_bound;
        std::vector<std::pair<double, double>> roof_points;
    };

    PerformanceBounds analyzeBounds();

private:
    RooflineAnalyzer() = default;

    // internal state
    RooflineConfig config_;
    std::vector<RooflinePoint> data_points_;

    // analysis functions
    double computeRoofline(double arithmetic_intensity);
    void updateBounds();
};

// Timeline Analyzer
class TimelineAnalyzer {
public:
    static TimelineAnalyzer& getInstance();

    // Event types
    enum class TimelineEventType {
        KERNEL_LAUNCH,
        MEMORY_TRANSFER,
        SYNCHRONIZATION,
        COMPUTATION,
        COMMUNICATION
    };

    // Timeline event
    struct TimelineEvent {
        TimelineEventType type;
        std::string name;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
        int device_id;
        std::string additional_info;
    };

    // log event
    void recordEvent(const TimelineEvent& event);

    // timeline analysis
    struct TimelineAnalysis {
        double total_duration;
        double compute_time;
        double memory_time;
        double sync_time;
        double idle_time;
        std::vector<std::pair<std::string, double>> hotspots;
    };

    TimelineAnalysis analyzeTimeline();

    // visualization
    void generateVisualization(const std::string& output_file);

private:
    TimelineAnalyzer() = default;

    // internal state
    std::vector<TimelineEvent> events_;

    // analysis functions
    void identifyHotspots();
    void analyzeOverlap();
    void generateGanttChart(const std::string& output_file);
};

} // namespace profiler
} // namespace uta
