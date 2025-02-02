#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "bottleneck_detector.hpp"

namespace uta {
namespace profiler {

// visualization type
enum class VisualizationType {
    TIMELINE,           // timeline view
    ROOFLINE,          // Roofline model
    HEATMAP,           // heatmap
    FLAMEGRAPH,        // flamegraph
    DEPENDENCY_GRAPH   // dependency graph
};

// visualization configuration
struct VisualizationConfig {
    std::string output_format;   // output format (HTML, SVG, PNG)
    std::string theme;           // theme style
    bool interactive;            // interactive mode
    bool show_tooltips;          // Whether to display tooltips
    std::string custom_css;      // custom CSS
};

// performance visualizer
class PerformanceVisualizer {
public:
    static PerformanceVisualizer& getInstance();

    // initialization
    void initialize(const VisualizationConfig& config);

    // visualization generation
    void generateVisualization(
        VisualizationType type,
        const std::string& output_file,
        const std::string& title = ""
    );

    // data source configuration
    void setTimelineData(const std::vector<TimelineAnalyzer::TimelineEvent>& events);
    void setRooflineData(const std::vector<RooflineAnalyzer::RooflinePoint>& points);
    void setBottleneckData(const BottleneckDetector::PerformanceAnalysis& analysis);

    // interactive controls
    struct InteractiveControls {
        bool enable_zooming;
        bool enable_filtering;
        bool enable_selection;
        bool enable_animation;
    };

    void setInteractiveControls(const InteractiveControls& controls);

private:
    PerformanceVisualizer() = default;

    // visualization generators
    void generateTimeline(const std::string& output_file);
    void generateRoofline(const std::string& output_file);
    void generateHeatmap(const std::string& output_file);
    void generateFlamegraph(const std::string& output_file);
    void generateDependencyGraph(const std::string& output_file);

    // style manager
    struct StyleManager {
        std::string getColorScheme();
        std::string getLayoutTemplate();
        std::string generateCSS();
    };

    // internal state
    VisualizationConfig config_;
    InteractiveControls controls_;
    StyleManager style_manager_;

    // data cache
    std::vector<TimelineAnalyzer::TimelineEvent> timeline_data_;
    std::vector<RooflineAnalyzer::RooflinePoint> roofline_data_;
    BottleneckDetector::PerformanceAnalysis bottleneck_data_;
};

// flame graph generator
class FlameGraphGenerator {
public:
    static FlameGraphGenerator& getInstance();

    // configuration
    struct FlameGraphConfig {
        bool show_timestamps;
        bool show_percentages;
        bool inverted;
        std::string color_scheme;
    };

    void setConfig(const FlameGraphConfig& config);

    // Data collection
    struct StackFrame {
        std::string name;
        std::string category;
        double duration;
        std::vector<StackFrame> children;
    };

    void addStackTrace(const std::vector<StackFrame>& trace);

    // Generate a flame map
    void generateFlameGraph(const std::string& output_file);

private:
    FlameGraphGenerator() = default;

    // internal state
    FlameGraphConfig config_;
    std::vector<std::vector<StackFrame>> stack_traces_;

    // generate functions
    void aggregateStacks();
    void calculatePercentages();
    void generateSVG(const std::string& output_file);
};

// dependency graph generator
class DependencyGraphGenerator {
public:
    static DependencyGraphGenerator& getInstance();

    // Node type
    struct Node {
        std::string id;
        std::string label;
        std::string type;
        std::unordered_map<std::string, std::string> attributes;
    };

    // edge type
    struct Edge {
        std::string from;
        std::string to;
        std::string type;
        std::unordered_map<std::string, std::string> attributes;
    };

    // graphics configuration
    struct GraphConfig {
        std::string layout;      // layout algorithm
        bool show_weights;       // show weights
        bool cluster_nodes;      // cluster nodes
        std::string orientation; // graph orientation
    };

    void setConfig(const GraphConfig& config);

    // graph construction
    void addNode(const Node& node);
    void addEdge(const Edge& edge);
    void clear();

    // generate graph
    void generateGraph(const std::string& output_file);

private:
    DependencyGraphGenerator() = default;

    // internal state
    GraphConfig config_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;

    // generator function
    void layoutGraph();
    void applyStyles();
    void exportGraph(const std::string& output_file);
};

} // namespace profiler
} // namespace uta
