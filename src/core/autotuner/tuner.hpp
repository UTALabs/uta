#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "../profiler/kernel_profiler.hpp"

namespace uta {
namespace autotuner {

struct TuningParameter {
    std::string name;
    std::vector<int64_t> possible_values;
    int64_t current_value;
    double impact_score;  // Estimated impact on performance
};

struct KernelConfig {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_memory_size;
    std::vector<TuningParameter> parameters;
};

struct TuningResult {
    KernelConfig best_config;
    double performance_metric;  // FLOPS, latency, etc.
    std::string optimization_log;
};

class AutoTuner {
public:
    static AutoTuner& getInstance();

    // Initialize tuning session
    void initTuning(const std::string& kernel_name,
                   const std::vector<TuningParameter>& params);

    // Define objective function
    using ObjectiveFunction = std::function<double(const KernelConfig&)>;
    void setObjectiveFunction(ObjectiveFunction func);

    // Tuning methods
    TuningResult tuneGridBlock();
    TuningResult tuneMemoryAccess();
    TuningResult tuneInstructionSchedule();
    
    // Advanced tuning features
    struct TuningConstraints {
        size_t max_shared_memory;
        size_t max_registers_per_thread;
        size_t min_occupancy;
    };
    
    void setConstraints(const TuningConstraints& constraints);
    
    // Genetic algorithm parameters
    struct GeneticParams {
        size_t population_size;
        size_t num_generations;
        float mutation_rate;
        float crossover_rate;
    };
    
    void setGeneticParams(const GeneticParams& params);

private:
    AutoTuner() = default;
    
    // Internal tuning algorithms
    KernelConfig geneticSearch();
    KernelConfig bayesianOptimization();
    KernelConfig randomSearch();
    
    // Helper functions
    bool validateConfig(const KernelConfig& config);
    double evaluateConfig(const KernelConfig& config);
    std::vector<KernelConfig> generateCandidates();
    
    // Internal state
    std::string current_kernel_;
    std::vector<TuningParameter> parameters_;
    ObjectiveFunction objective_function_;
    TuningConstraints constraints_;
    GeneticParams genetic_params_;
};

} // namespace autotuner
} // namespace uta
